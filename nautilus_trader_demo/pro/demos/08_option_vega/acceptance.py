"""Offline review of an actual 08 Recording report; does not connect or place orders."""
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo


def review_recording(summary, *, trading_day, option_slot="13:58", require_target=True):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    require(summary.get("status") == "passed", "运行本身未通过")
    require(summary.get("mode") == "recording" and summary.get("execution_origin") == "recording", "不是Recording来源")
    require(summary.get("simnow_environment") == "realtime", "真实实时验收不能使用Replay")
    require(summary.get("market_source") == "ctp" and summary.get("index_source") == "dolphindb-dfs", "行情来源不符合CTP+DFS验收范围")
    require(summary.get("trading_day") == trading_day and summary.get("valuation_day") == trading_day, "交易日/估值日不符")
    require(summary.get("observed_md_trading_day") == trading_day, "没有观测到当天CTP MD交易日")
    for key, label in (("orders_submitted", "柜台报单"), ("fills_received", "成交")):
        if key not in summary:
            errors.append(f"报告缺少{label}计数证据")
        else:
            require(summary[key] == 0, f"Recording出现{label}")
    require(summary.get("account_positions") is None and summary.get("observed_td_trading_day") is None, "Recording关联了物理账户")
    require(summary.get("calendar_manifest", {}).get("backend") == "compiled", "Live没有使用公共编译日历")
    queries = summary.get("feature_queries", [])
    # Reports are written after stop: connected=False is normal, retain query evidence.
    query_ok = bool(queries) and all(q.get("last_query_ns") is not None and not q.get("failure")
        and q.get("cached_records", 0) > 0 for q in queries)
    delivered = query_ok and all(q.get("emitted_versions") for q in queries)
    require(query_ok, "指数查询无成功证据")
    require(delivered, "指数尚未送达策略分钟帧")
    require(not summary.get("cleanup_errors"), "资源清理失败")
    entries = [row for row in summary.get("entry_audit", [])
        if row.get("date") == datetime.strptime(trading_day, "%Y%m%d").date().isoformat()
        and row.get("slot") == option_slot]
    require(len(entries) == 1, "选约时点没有唯一决策审计")
    frames = {row["valuation_ns"]: row for row in summary.get("frame_audit", [])}
    target_verified = False
    if len(entries) == 1:
        entry = entries[0]
        stamp = entry["signal_ns"]
        slot = datetime.fromtimestamp(stamp/1e9, ZoneInfo("Asia/Shanghai")).strftime("%H:%M")
        require(slot == option_slot, "决策实际业务时间不符")
        frame = frames.get(stamp, {})
        require(frame.get("status") == "CONSUMED", "选约帧没有实际消费")
        observations = frame.get("observations", {})
        index_key = summary.get("index_code", "000852")
        require(index_key in observations, "选约帧缺指数")
        require(observations.get(index_key, {}).get("source_id") == "index-bars", "指数来源绑定不符")
        require(all(row.get("source_ns") in (stamp, stamp-1) and
            row.get("received_ns", -1) >= stamp and row.get("received_ns", -1) <= frame.get("deadline_ns", -1)
            for row in observations.values()), "选约帧业务/接收时间不一致")
        require(frame.get("request_ns", -1) >= frame.get("deadline_ns", 0), "分钟帧未等到接收期限")
        targets = {key: value for key, value in entry.get("targets", {}).items() if Decimal(str(value)) != 0}
        require(index_key not in targets, "指数进入交易目标")
        require(all(key in observations for key in targets), "目标腿没有同分钟行情")
        require(all(observations.get(key, {}).get("source_id") == "market-bars" for key in targets), "交易腿来源绑定不符")
        require(any(key.startswith("IM") for key in observations), "选约帧缺同月期货")
        require(all(entry.get("future_bindings", {}).get(key) in observations for key in targets), "目标期权缺明确绑定的同月期货行情")
        require(entry.get("outcome") in {"TARGET", "NO_VALID_ENTRY", "ZERO_ALLOCATION"}, "选约未完成评价")
        recorded = [request for request in summary.get("recorded_requests", []) if request.get("signal_ns") == stamp]
        actual_targets = [{key.rsplit(".", 1)[0]: Decimal(str(value)) for key, value in request.get("targets", {}).items()
            if Decimal(str(value)) != 0} for request in recorded]
        matched = any(all(values.get(key) == Decimal(str(value)) for key, value in targets.items()) for values in actual_targets)
        target_verified = entry.get("outcome") == "TARGET" and bool(targets) and matched
        require(all(summary.get("index_code", "000852") not in values for values in actual_targets), "实际Recording请求包含指数目标")
    if require_target:
        require(target_verified, "没有非零Recording目标，目标链验收仍未通过")
    skipped = [row for row in frames.values() if row.get("status") == "SKIPPED_INDEX"]
    require(not any(entry.get("signal_ns") == row["valuation_ns"] for entry in entries for row in skipped), "超时跳过帧产生了选约决策")
    return {"status": "failed" if errors else "passed", "scope": "target_chain" if require_target else "decision_only",
        "trading_day": trading_day, "option_slot": option_slot, "errors": errors,
        "target_chain_verified": target_verified, "entry_audit": entries,
        "runtime_failure": summary.get("failure"), "session_readiness": summary.get("session_readiness"),
        "index_query_verified": query_ok, "index_bar_delivery_verified": bool(delivered),
        "frames": summary.get("frames"), "skipped_index_frames": summary.get("skipped_index_frames"),
        "timeout_seen_online": bool(skipped), "timeout_note": "已观测同分钟指数缺失后跳过" if skipped else "无线上超时样本时，超时拒绝仅由离线测试覆盖"}
