"""外部刷卡批次合并预览：夏令时（DST）时间差口径回归测试。

覆盖修复的缺陷：
1. 候选筛选用 ``datetime.timestamp()``（按服务进程时区解释朴素时刻），
   返回的有符号秒差却用记录时刻直接相减，两者在实行夏令时的时区不一致。
   America/New_York 为例：
   - 2026-03-08 春季跳钟：01:59:30 与 03:00:00 的记录时刻相差
     60 分 30 秒（> 120 秒容差），但时间戳只差 30 秒，旧实现误配为重复；
   - 2026-11-01 秋季回拨：01:59:30 与 02:00:00 的记录时刻相差
     30 秒（在 30 秒容差内），但时间戳相差 60 分 30 秒，旧实现漏配。
2. 修复后候选筛选、全局一对一匹配费用与返回的绝对 / 有符号秒差统一采用
   「与输入文本一致的记录时刻字面值直接求差」语义，结果与服务时区无关，
   跨夏令时（春季 / 秋季）与普通日期稳定；容差按闭区间判定。
3. 仍然先最大化配对数、再最小化总绝对秒差，同分时按基准 / 导入行号
   破同分；去重合并文本、请求与响应字段、只读行为保持兼容。

测试在 America/New_York 与 UTC 两套进程时区下重复同一组断言
（tzset 仅影响朴素 datetime.timestamp() 的解释，修复后不应产生差异）。

运行：``python3 -m pytest tests/ -q``（依赖见 requirements.txt，
另需 pytest 与 httpx）。
"""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from app import db, main
from app.core import external_merge_preview, parse_records_text

SPRING_A = "A001, 2026-03-08 01:59:30, 科技园, 进"
SPRING_B = "A001, 2026-03-08 03:00:00, 科技园, 进"
SPRING_EDGE = "A001, 2026-03-08 02:01:30, 科技园, 进"
FALL_A = "A001, 2026-11-01 01:59:30, 科技园, 进"
FALL_B = "A001, 2026-11-01 02:00:00, 科技园, 进"
NORMAL_A = "A001, 2026-09-18 08:00:00, 科技园, 进"
NORMAL_B = "A001, 2026-09-18 08:00:03, 科技园, 进"


def _records(text: str):
    records, errors = parse_records_text(text + "\n")
    assert not errors, errors
    return records


@pytest.fixture(params=["America/New_York", "UTC"])
def process_tz(request, monkeypatch):
    """把进程时区切到指定值；测试结束恢复原环境（tzset 全局生效）。"""
    old_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", request.param)
    if hasattr(time, "tzset"):
        time.tzset()
    yield request.param
    if old_tz is None:
        monkeypatch.delenv("TZ", raising=False)
    else:
        monkeypatch.setenv("TZ", old_tz)
    if hasattr(time, "tzset"):
        time.tzset()


# ---------------------------------------------------------------------------
# 核心层
# ---------------------------------------------------------------------------

def test_spring_forward_not_matched_over_tolerance(process_tz):
    # 记录时刻差 3630 秒，120 秒容差内不得配成重复（旧实现误配）。
    r = external_merge_preview(_records(SPRING_A), _records(SPRING_B), 120)
    assert r.ok
    assert r.pair_count == 0
    assert len(r.base_only) == 1
    assert len(r.import_only) == 1
    assert r.total_abs_diff_seconds == 0


def test_spring_forward_boundary_closed_interval(process_tz):
    # 01:59:30 -> 02:01:30 记录时刻恰为 120 秒：闭区间边界仍算候选。
    r = external_merge_preview(_records(SPRING_A), _records(SPRING_EDGE), 120)
    assert r.pair_count == 1
    assert r.total_abs_diff_seconds == 120
    assert r.pairs[0]["abs_diff_seconds"] == 120
    assert r.pairs[0]["delta_seconds"] == 120
    # 容差 119 秒则不配对。
    r2 = external_merge_preview(_records(SPRING_A), _records(SPRING_EDGE), 119)
    assert r2.pair_count == 0


def test_fall_back_matched_within_tolerance(process_tz):
    # 记录时刻差 30 秒：30 秒容差内必须配对（旧实现漏配）。
    r = external_merge_preview(_records(FALL_A), _records(FALL_B), 30)
    assert r.pair_count == 1
    assert r.total_abs_diff_seconds == 30
    p = r.pairs[0]
    assert p["abs_diff_seconds"] == 30
    assert p["delta_seconds"] == 30  # 导入 − 基准，导入更晚
    # 反向放置时绝对差不变、有符号差取负。
    r2 = external_merge_preview(_records(FALL_B), _records(FALL_A), 30)
    assert r2.pair_count == 1
    assert r2.pairs[0]["abs_diff_seconds"] == 30
    assert r2.pairs[0]["delta_seconds"] == -30


def test_spring_forward_max_cardinality_then_min_cost(process_tz):
    # 两侧各两条：字面值口径下
    #   01:59:30↔02:00:00 = 30 秒，03:00:00↔03:00:00 = 0 秒，
    # 可配成两对（最大化配对数），总绝对差 30 秒；
    # 旧时间戳口径只能配一对（01:59:30↔03:00:00）。
    base = _records("\n".join([SPRING_A, SPRING_B]))
    imports = _records("\n".join([
        "A001, 2026-03-08 02:00:00, 科技园, 进", SPRING_B]))
    r = external_merge_preview(base, imports, 120)
    assert r.pair_count == 2
    assert r.total_abs_diff_seconds == 30
    by_import = {p["import_time"]: p for p in r.pairs}
    assert by_import["2026-03-08 03:00:00"]["base_time"] == "2026-03-08 03:00:00"
    assert by_import["2026-03-08 03:00:00"]["abs_diff_seconds"] == 0
    assert by_import["2026-03-08 02:00:00"]["base_time"] == "2026-03-08 01:59:30"
    assert by_import["2026-03-08 02:00:00"]["abs_diff_seconds"] == 30
    assert not r.base_only and not r.import_only
    # 合并文本：配对取基准行，按记录时刻排序（仅两条基准记录）。
    assert r.merged_text.splitlines() == [SPRING_A, SPRING_B]


def test_normal_date_and_zero_tolerance(process_tz):
    r = external_merge_preview(_records(NORMAL_A), _records(NORMAL_B), 30)
    assert r.pair_count == 1
    assert r.pairs[0]["abs_diff_seconds"] == 3
    assert r.pairs[0]["delta_seconds"] == 3
    # 容差 0 为闭区间：必须字面值完全相同。
    r0 = external_merge_preview(_records(NORMAL_A), _records(NORMAL_B), 0)
    assert r0.pair_count == 0
    r0same = external_merge_preview(_records(NORMAL_A), _records(NORMAL_A), 0)
    assert r0same.pair_count == 1
    assert r0same.pairs[0]["abs_diff_seconds"] == 0


def test_line_number_tie_break(process_tz):
    # 两组配对总绝对差相同时，按基准行号和、再按导入行号和破同分。
    base = _records(
        "A001, 2026-09-18 08:00:00, 科技园, 进\n"
        "A001, 2026-09-18 08:00:05, 科技园, 进\n")
    imports = _records(
        "A001, 2026-09-18 08:00:00, 科技园, 进\n"
        "A001, 2026-09-18 08:00:05, 科技园, 进\n")
    r = external_merge_preview(base, imports, 120)
    got = sorted((p["base"]["line"], p["import"]["line"]) for p in r.pairs)
    assert got == [(1, 1), (2, 2)]


def test_invalid_tolerance_still_config_bad(process_tz):
    r = external_merge_preview(_records(NORMAL_A), _records(NORMAL_B), 121)
    assert not r.ok
    assert r.errors and r.errors[0]["code"] == "CONFIG_BAD"


# ---------------------------------------------------------------------------
# HTTP API（端到端：已保存基准记录 + 粘贴导入文本）
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = str(tmp_path / "bus-test.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    return path


def _client() -> TestClient:
    return TestClient(main.app)


def test_http_spring_and_fall_preview(process_tz, temp_db):
    client = _client()
    # 基准库各放一条春季 / 秋季记录（不需要站点 / 票价：合并预览不依赖）。
    base_text = "\n".join([SPRING_A, FALL_A]) + "\n"
    resp = client.put("/api/records", json={"text": base_text})
    assert resp.status_code == 200

    # 春季：记录时刻相差 60 分 30 秒，120 秒容差内不得配对。
    resp = client.post(
        "/api/external-merge-preview",
        json={"text": SPRING_B + "\n", "tolerance_seconds": 120})
    data = resp.json()
    assert resp.status_code == 200, data
    assert data["ok"] is True
    assert data["pair_count"] == 0
    assert data["base_only_count"] == 2
    assert data["import_only_count"] == 1
    assert data["total_abs_diff_seconds"] == 0
    # 字段形态保持兼容
    assert set(data) == {
        "ok", "tolerance_seconds", "base_count", "import_count",
        "pair_count", "base_only_count", "import_only_count",
        "merged_count", "total_abs_diff_seconds", "pairs",
        "base_only", "import_only", "merged_text"}
    # 去重合并文本：两侧记录都保留，按记录时刻排序（03-08 早于 11-01）
    assert data["merged_text"].splitlines() == [
        SPRING_A, SPRING_B, FALL_A]

    # 秋季：记录时刻相差 30 秒，30 秒容差必须配对。
    resp = client.post(
        "/api/external-merge-preview",
        json={"text": FALL_B + "\n", "tolerance_seconds": 30})
    data = resp.json()
    assert resp.status_code == 200, data
    assert data["pair_count"] == 1
    assert data["base_only_count"] == 1
    assert data["import_only_count"] == 0
    assert data["merged_count"] == 2
    assert data["total_abs_diff_seconds"] == 30
    p = data["pairs"][0]
    assert p["abs_diff_seconds"] == 30
    assert p["delta_seconds"] == 30
    assert p["base_time"] == "2026-11-01 01:59:30"
    assert p["import_time"] == "2026-11-01 02:00:00"
    # 配对取基准行计入合并文本（导入行不重复出现）
    assert data["merged_text"].splitlines() == [SPRING_A, FALL_A]


def test_http_tolerance_boundary_and_readonly(process_tz, temp_db):
    client = _client()
    resp = client.put("/api/records", json={"text": SPRING_A + "\n"})
    assert resp.status_code == 200

    # 恰好 120 秒（闭区间）配对；119 秒不配对。
    resp = client.post(
        "/api/external-merge-preview",
        json={"text": SPRING_EDGE + "\n", "tolerance_seconds": 120})
    assert resp.json()["pair_count"] == 1
    resp = client.post(
        "/api/external-merge-preview",
        json={"text": SPRING_EDGE + "\n", "tolerance_seconds": 119})
    assert resp.json()["pair_count"] == 0

    # 非法容差：422 + CONFIG_BAD
    resp = client.post(
        "/api/external-merge-preview",
        json={"text": SPRING_B + "\n", "tolerance_seconds": 121})
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["code"] == "CONFIG_BAD"

    # 只读：已保存记录不被修改、导入记录不入库。
    resp = client.get("/api/records")
    got = client.get("/api/records").json()["items"]
    assert len(got) == 1
    assert got[0]["time"] == "2026-03-08 01:59:30"

    # 待导入文本解析错误沿用 RECORD_BAD_*（行号指向粘贴文本）。
    resp = client.post(
        "/api/external-merge-preview",
        json={"text": "A001, not-a-time, 科技园, 进\n",
              "tolerance_seconds": 30})
    assert resp.status_code == 422
    err = resp.json()["errors"][0]
    assert err["code"] == "RECORD_BAD_TIME"
    assert err["location"]["line"] == 1


def test_endpoint_schema_fields(monkeypatch, temp_db):
    """端点层透传：MergePreviewIn 字段与既有请求形态保持兼容。"""
    client = _client()
    resp = client.put("/api/records", json={"text": FALL_A + "\n"})
    assert resp.status_code == 200
    resp = client.post(
        "/api/external-merge-preview",
        json={"text": FALL_B + "\n", "tolerance_seconds": 30})
    data = resp.json()
    assert data["ok"] is True
    assert data["pair_count"] == 1
