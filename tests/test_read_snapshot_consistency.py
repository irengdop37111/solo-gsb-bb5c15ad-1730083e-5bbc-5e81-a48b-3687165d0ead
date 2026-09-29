"""只读接口的并发混读回归测试：每次请求的多次读取必须来自同一已提交快照。

保存 / 核验在**同一事务**内整体替换账单并清除核验时间（争议预演还要
替换配置、站点、票价与刷卡记录）。若只读接口把多次读取拆成各自独立的
自动提交 SELECT，则可能在两次读取之间恰好插入这样的提交，得到
「有行程却无核验时间」、或把两个提交版本的账单 / 输入拼在一起的混合
响应。

这里用线程在目标读取函数**第一次**执行之后立即启动一个并发保存
（清空账单、清除核验时间、递增数据版本，同一写事务提交），并等待一个
小延时让写者运行到提交的锁等待点；同步最终依赖 SQLite 自身的锁：

- 旧实现（无显式快照）：第一次读取所在语句结束即释放 SHARED 锁，
  写者立刻拿到 RESERVED/EXCLUSIVE 锁并提交，随后的第二次读取看到
  新版本 -> 响应混合两版（下列「变化前」断言会失败）；
- 新实现（显式 ``BEGIN`` 一致快照）：读事务在读完前一直持有 SHARED
  锁，写者的提交必须等待读者 ``rollback()`` 释放快照；于是响应完整
  对应变化前一版；下一次请求才完整对应变化后一版。

运行：``python3 -m pytest tests/ -q``。
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


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """每个用例一个独立数据库文件（db.connect 默认读取模块级 DB_PATH）。"""
    path = str(tmp_path / "bus-test.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    main.put_stations(TextIn(text=STATIONS))
    main.put_fares(TextIn(text=FARES))
    main.put_records(TextIn(text=RECORDS))
    resp = main.run_verify()
    assert not isinstance(resp, JSONResponse) and resp["ok"] is True
    return path


def _payload(resp) -> dict:
    if isinstance(resp, JSONResponse):
        return json.loads(resp.body.decode("utf-8"))
    return resp


def install_writer_after_first_read(monkeypatch, target, write_action,
                                    settle=0.15):
    """包装 db 层读取函数：第一次执行后立刻在另一线程启动并发写。

    返回 committed 事件（写事务已提交）。旧实现下该提交早于请求的
    第二次读取（响应混合两版）；新实现下晚于请求 rollback 释放快照
    （响应完整对应变化前一版）。``settle`` 让写者先运行到提交的锁
    等待点，保证它与请求后续读取真实并发。
    """
    committed = threading.Event()
    fired = threading.Event()
    lock = threading.Lock()
    real_target = target

    def wrapped(*args, **kwargs):
        result = real_target(*args, **kwargs)
        with lock:
            first = not fired.is_set()
            if first:
                fired.set()

                def _run():
                    write_action()
                    committed.set()

                th = threading.Thread(target=_run, daemon=True)
                th.start()
                time.sleep(settle)  # 写者运行到锁等待点
        return result

    monkeypatch.setattr(db, target.__name__, wrapped)
    return committed


def _concurrent_clear_records():
    """并发保存：整批替换为同一批记录（版本递增、账单与核验时间清空）。"""
    main.put_records(TextIn(text=RECORDS))


# ---------------------------------------------------------------------------
# GET /api/bills：账单与核验时间必须同版（不得「有行程却无核验时间」）
# ---------------------------------------------------------------------------

def test_bills_response_is_one_committed_version(temp_db, monkeypatch):
    committed = install_writer_after_first_read(
        monkeypatch, db.list_bills, _concurrent_clear_records)

    resp = main.get_bills()

    # 响应完整对应变化前一版：1 条行程且带有核验时间（不得混合）
    assert resp["count"] == 1
    assert len(resp["items"]) == 1
    assert resp["last_verified_at"] is not None
    # 快照持读锁到读完：写者在请求结束后才能提交
    assert committed.wait(5.0)

    # 下一次请求完整对应变化后一版：账单与核验时间都已清空
    after = main.get_bills()
    assert after["count"] == 0
    assert after["items"] == []
    assert after["last_verified_at"] is None


# ---------------------------------------------------------------------------
# POST /api/pass-backtest：账单与核验时间同快照（NO_BILLS 判定口径一致）
# ---------------------------------------------------------------------------

def test_pass_backtest_uses_one_snapshot(temp_db, monkeypatch):
    committed = install_writer_after_first_read(
        monkeypatch, db.list_bills, _concurrent_clear_records)

    body = PassBacktestIn(passes_text="一日票, 6.00, 1\n",
                          start_day="2026-09-18", end_day="2026-09-18")
    resp = main.run_pass_backtest(body)

    # 变化前版本：账单有效，回测成功（A001 当日实扣 3 元）
    assert not isinstance(resp, JSONResponse)
    assert resp["ok"] is True
    assert resp["last_verified_at"] is not None
    assert resp["trip_count"] == 1
    assert committed.wait(5.0)

    # 变化后版本：账单已清空 -> 422 NO_BILLS
    after = main.run_pass_backtest(body)
    assert isinstance(after, JSONResponse) and after.status_code == 422
    assert _payload(after)["errors"][0]["code"] == "NO_BILLS"


# ---------------------------------------------------------------------------
# POST /api/dispute-preview：配置/站点/票价/记录/账单/核验时间同快照
# ---------------------------------------------------------------------------

def test_dispute_preview_uses_one_snapshot(temp_db, monkeypatch):
    # load_domain 是争议预演读取「站点 / 票价 / 记录」的入口（配置在它
    # 之前读取）：第一次返回后即插入并发保存，随后的账单 / 核验时间读取
    # 在新实现下仍处于同一快照，旧实现下会读到被清空后的新版本。
    committed = install_writer_after_first_read(
        monkeypatch, db.load_domain, _concurrent_clear_records)

    # 第 2 行（A001 出站，中心公园）临时改到科技园：更正有效、可出对比
    body = DisputePreviewIn(edits=[DisputeEditIn(line=2, station="科技园")])
    resp = main.run_dispute_preview(body)

    # 变化前版本：输入与账单同快照，预演成功并带回核验时间
    assert not isinstance(resp, JSONResponse)
    assert resp["ok"] is True
    assert resp["last_verified_at"] is not None
    assert committed.wait(5.0)

    # 变化后版本：账单与核验时间均已清空 -> NO_BILLS
    after = main.run_dispute_preview(body)
    assert isinstance(after, JSONResponse) and after.status_code == 422
    assert _payload(after)["errors"][0]["code"] == "NO_BILLS"


# ---------------------------------------------------------------------------
# POST /api/compensation-preview：账单与核验时间同快照（金额不混版）
# ---------------------------------------------------------------------------

def test_compensation_preview_uses_one_snapshot(temp_db, monkeypatch):
    committed = install_writer_after_first_read(
        monkeypatch, db.list_bills, _concurrent_clear_records)

    body = CompensationPreviewIn(
        station="科技园",
        start_time="2026-09-18 07:00:00",
        end_time="2026-09-18 09:00:00",
        budget=10, per_card_cap=10)
    resp = main.run_compensation_preview(body)

    # 变化前版本：A001 的科技园进站落在区间内 -> 1 程、实扣 3 元
    assert not isinstance(resp, JSONResponse)
    assert resp["ok"] is True
    assert resp["last_verified_at"] is not None
    assert resp["trip_count"] == 1
    assert resp["charged_total"] == 3.0
    assert committed.wait(5.0)

    # 变化后版本：账单已清空 -> 200 空结果（未分配金额等于预算全额）
    after = main.run_compensation_preview(body)
    assert not isinstance(after, JSONResponse)
    assert after["trip_count"] == 0
    assert after["cards"] == []
    assert after["last_verified_at"] is None
    assert after["unallocated"] == 10.0
