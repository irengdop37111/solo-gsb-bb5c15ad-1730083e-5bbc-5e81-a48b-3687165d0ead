"""只读接口「一致快照读取」并发回归测试。

修复前，账单查询、通票回测、账单争议更正预演与服务中断补偿预览分次
读取数据库（如先读账单、再读核验时间）；并发保存或核验恰好在两次读取
之间提交时，同一次响应会拼到两个不同的已提交版本：

- ``GET /api/bills``：有行程账单却没有核验时间（或反之）；
- ``/api/pass-backtest``、``/api/compensation-preview``：基于另一版本的
  账单计算，却带着不匹配的核验时间，金额 / NO_BILLS 判断错误；
- ``/api/dispute-preview``：当前账单来自旧记录、更正重算基于新记录
  （配置 / 站点 / 票价 / 刷卡记录也可能各属一版），差额失真。

修复后，每次请求在**同一个已提交快照**内读出全部所需数据，读完即释放
读锁，再进行只读计算；因此响应完整对应变化前或变化后的一版。

这里用线程 + 在「快照内最后一次读取」上设闸的方式确定性复现交错：
读线程持住读快照后，写线程开始保存——读线程持有 SHARED 锁期间，写线程
的提交只能阻塞，直到读线程的快照读取结束。

运行：``python3 -m pytest tests/ -q``（依赖见 requirements.txt，
另需 pytest 与 httpx）。
"""
from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.responses import JSONResponse

from app import db, main
from app.schemas import (
    CompensationPreviewIn,
    DisputeEditIn,
    DisputePreviewIn,
    PassBacktestIn,
    TextIn,
)

STATIONS = "科技园, 1\n中心公园, 2\n"
FARES = "1, 1, 2.00\n1, 2, 3.00\n2, 1, 3.00\n2, 2, 2.00\n"
RECORDS = (
    "A001, 2026-09-18 08:00:00, 科技园, 进\n"
    "A001, 2026-09-18 08:25:00, 中心公园, 出\n"
)

# 写线程在闸口等待的时间：足够让其走到提交并被 SHARED 锁挡住；
# 远小于连接 busy_timeout（5 秒），不会误触锁超时。
HOLD_SECONDS = 1.0
LAST_VERIFIED_KEY = "last_verified_at"


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """每个用例一个独立数据库文件（db.connect 默认读取模块级 DB_PATH）。"""
    path = str(tmp_path / "bus-test.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    return path


def _seed_and_verify() -> None:
    """写入基础数据并成功核验（生成 1 笔账单与核验时间）。"""
    assert not isinstance(main.put_stations(TextIn(text=STATIONS)), JSONResponse)
    assert not isinstance(main.put_fares(TextIn(text=FARES)), JSONResponse)
    assert not isinstance(main.put_records(TextIn(text=RECORDS)), JSONResponse)
    resp = main.run_verify()
    assert not isinstance(resp, JSONResponse)
    assert resp["ok"] is True and resp["bill_count"] == 1


def _payload(resp: JSONResponse) -> dict:
    return json.loads(resp.body.decode("utf-8"))


def _gate_last_verified_read(monkeypatch, writer):
    """在读取核验时间（快照内最后一次读取）处插入并发保存。

    包装 db.get_meta：当读线程读取 last_verified_at 时，启动写线程
    执行一次保存（同事务递增版本并清空账单与核验时间），随后保持
    HOLD_SECONDS 再放行本次读取；写线程在端点返回后再 join。

    - 修复后读线程处于显式读快照中（持 SHARED 锁），写线程的提交在
      快照释放前被阻塞，读线程必然读到变化前的一版；快照释放后
      写线程立刻完成提交。
    - 修复前各 SELECT 各自自动提交，写线程在读线程等待期间即可提交，
      读线程拿到「旧账单 + 新（空）核验时间」的混合响应。
    """
    real_get_meta = db.get_meta
    state = {"fired": False, "thread": None}

    def gated_get_meta(conn, key):
        if key == LAST_VERIFIED_KEY and not state["fired"]:
            state["fired"] = True
            t = threading.Thread(target=writer)
            state["thread"] = t
            t.start()
            # 让写线程走到提交点：修复后会在此被 SHARED 锁阻塞，
            # 修复前则直接提交完成。不能在此 join——在快照内 join 会把
            # 阻塞中的写线程一直等到 busy_timeout 锁超时。
            time.sleep(HOLD_SECONDS)
        return real_get_meta(conn, key)

    monkeypatch.setattr(db, "get_meta", gated_get_meta)
    return state


def _concurrent_save():
    """并发请求：重新保存票价（同事务递增版本、清空账单与核验时间）。"""
    resp = main.put_fares(TextIn(text=FARES))
    assert not isinstance(resp, JSONResponse)


def _run_in_thread(target):
    """在工作线程中运行同步端点处理器，返回 (结果, 异常)。"""
    box: dict = {}

    def worker():
        try:
            box["result"] = target()
        except BaseException as ex:  # 读线程内任何异常都回传断言
            box["exc"] = ex

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return box.get("result"), box.get("exc")


# ---------------------------------------------------------------------------
# db.snapshot：跨连接的一致快照保证（无时序依赖）
# ---------------------------------------------------------------------------

def test_snapshot_keeps_one_committed_version(temp_db):
    """读快照期间另一连接提交清空：快照内仍是旧版本，释放后才见新版本。"""
    _seed_and_verify()

    def concurrent_clear():
        # 另一连接（等效另一请求 / 进程）：保存后清空账单与核验时间
        with db.connect() as other:
            db.clear_bills(other)

    with db.connect() as conn:
        with db.snapshot(conn):
            assert len(db.list_bills(conn)) == 1
            assert db.get_meta(conn, LAST_VERIFIED_KEY) is not None

            # 写线程的提交在读线程持有 SHARED 锁期间被阻塞（同线程内
            # 再开写连接会自锁到 busy_timeout，故必须用线程）。
            t = threading.Thread(target=concurrent_clear)
            t.start()
            time.sleep(HOLD_SECONDS)

            # 同一快照内多次读取仍完整对应变化前的一版
            assert len(db.list_bills(conn)) == 1
            assert db.get_meta(conn, LAST_VERIFIED_KEY) is not None
        # 快照在此释放（rollback），写线程随即完成提交
        t.join()

    # 快照已释放：新连接看到变化后的版本（账单 / 核验时间均清空）
    with db.connect() as conn:
        assert db.list_bills(conn) == []
        assert db.get_meta(conn, LAST_VERIFIED_KEY) is None


# ---------------------------------------------------------------------------
# GET /api/bills：账单与核验时间来自同一快照
# ---------------------------------------------------------------------------

def test_get_bills_not_mixed_under_concurrent_save(temp_db, monkeypatch):
    _seed_and_verify()
    gate = _gate_last_verified_read(monkeypatch, _concurrent_save)

    result, exc = _run_in_thread(main.get_bills)
    gate["thread"].join()  # 快照释放后写线程立即完成提交
    assert exc is None
    # 响应完整对应变化前的一版：有 1 笔行程且核验时间仍在
    assert result["count"] == 1
    assert result["last_verified_at"] is not None

    # 读响应之后写入才生效：当前状态已是变化后的一版（账单清空）
    state = main.get_bills()
    assert state["count"] == 0
    assert state["last_verified_at"] is None


# ---------------------------------------------------------------------------
# POST /api/pass-backtest：账单 + 核验时间来自同一快照
# ---------------------------------------------------------------------------

def test_pass_backtest_not_mixed_under_concurrent_save(temp_db, monkeypatch):
    _seed_and_verify()
    _gate_last_verified_read(monkeypatch, _concurrent_save)

    body = PassBacktestIn(
        passes_text="一日票, 6.00, 1",
        start_day="2026-09-18", end_day="2026-09-18")
    gate = _gate_last_verified_read(monkeypatch, _concurrent_save)
    result, exc = _run_in_thread(lambda: main.run_pass_backtest(body))
    gate["thread"].join()
    assert exc is None

    # 混读时会出现「账单已清空 → NO_BILLS」或核验时间错配；
    # 修复后基于变化前版本正常回测：A001 当日实扣 3.00 元。
    assert not isinstance(result, JSONResponse), _payload(result)
    assert result["ok"] is True
    assert result["last_verified_at"] is not None
    assert result["card_count"] == 1
    assert result["cards"][0]["card"] == "A001"
    assert result["cards"][0]["baseline_total"] == 3.0


# ---------------------------------------------------------------------------
# POST /api/compensation-preview：账单 + 核验时间来自同一快照
# ---------------------------------------------------------------------------

def test_compensation_preview_not_mixed_under_concurrent_save(
        temp_db, monkeypatch):
    _seed_and_verify()
    _gate_last_verified_read(monkeypatch, _concurrent_save)

    body = CompensationPreviewIn(
        station="科技园",
        start_time="2026-09-18 07:56:00",
        end_time="2026-09-18 08:10:00",
        budget=12.0, per_card_cap=2.0)
    gate = _gate_last_verified_read(monkeypatch, _concurrent_save)
    result, exc = _run_in_thread(lambda: main.run_compensation_preview(body))
    gate["thread"].join()
    assert exc is None

    # 修复后响应完整对应变化前版本：A001 科技园进站命中，实扣 3.00、
    # 补偿封顶 2.00；混读时账单侧可能已为空导致结果与核验时间错配。
    assert not isinstance(result, JSONResponse), _payload(result)
    assert result["ok"] is True
    assert result["last_verified_at"] is not None
    assert result["trip_count"] == 1
    assert result["card_count"] == 1
    assert result["cards"][0]["card"] == "A001"
    assert result["cards"][0]["charged_total"] == 3.0
    assert result["cards"][0]["compensation"] == 2.0


# ---------------------------------------------------------------------------
# POST /api/dispute-preview：配置/站点/票价/记录/账单/核验时间同一快照
# ---------------------------------------------------------------------------

def test_dispute_preview_not_mixed_under_concurrent_save(temp_db, monkeypatch):
    _seed_and_verify()
    _gate_last_verified_read(monkeypatch, _concurrent_save)

    # 把第 2 行（A001 出站 中心公园，2 区）改为 科技园（1 区）：
    # 更正后为同区 1→1 行程，基础票价 3.00 → 2.00，应退款 1.00 元。
    body = DisputePreviewIn(edits=[DisputeEditIn(line=2, station="科技园")])
    gate = _gate_last_verified_read(monkeypatch, _concurrent_save)
    result, exc = _run_in_thread(lambda: main.run_dispute_preview(body))
    gate["thread"].join()
    assert exc is None

    assert not isinstance(result, JSONResponse), _payload(result)
    assert result["ok"] is True
    assert result["card"] == "A001"
    assert result["last_verified_at"] is not None
    # 当前账单与更正重算必须基于同一版输入：当前 3.00、更正 2.00、
    # 差额 -1.00（应退款）。混版时两侧会来自不同提交，差额无意义。
    assert result["summary"]["current_total"] == 3.0
    assert result["summary"]["corrected_total"] == 2.0
    assert result["summary"]["delta"] == -1.0
    assert result["summary"]["kind"] == "refund"


# ---------------------------------------------------------------------------
# 无并发交错时行为不变
# ---------------------------------------------------------------------------

def test_read_endpoints_unchanged_without_interleaving(temp_db):
    _seed_and_verify()

    bills = main.get_bills()
    assert bills["count"] == 1 and bills["last_verified_at"] is not None

    backtest = main.run_pass_backtest(PassBacktestIn(
        passes_text="一日票, 6.00, 1",
        start_day="2026-09-18", end_day="2026-09-18"))
    assert not isinstance(backtest, JSONResponse) and backtest["ok"] is True

    compensation = main.run_compensation_preview(CompensationPreviewIn(
        station="科技园",
        start_time="2026-09-18 07:56:00", end_time="2026-09-18 08:10:00",
        budget=12.0, per_card_cap=2.0))
    assert not isinstance(compensation, JSONResponse)
    assert compensation["card_count"] == 1

    dispute = main.run_dispute_preview(DisputePreviewIn(
        edits=[DisputeEditIn(line=2, station="科技园")]))
    assert not isinstance(dispute, JSONResponse)
    assert dispute["summary"]["delta"] == -1.0
