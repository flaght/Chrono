"""同一固定IM策略先保存退出、再启动恢复；只读数据库，目标仅内存捕获。"""
import argparse
from importlib import import_module
import json
from pathlib import Path
import uuid
from types import SimpleNamespace

from dotenv import load_dotenv

from bomber.framework.dataprep.history import HistoryError, HistoryUnavailable, ImDayWindow
from bomber.framework.market.basic.base import InstrumentId, make_bar
from .readonly import write_report
from .resume import digest
from .warmup import parse_args as parse_warmup_args

# 每次Python启动生成标识，报告不把同进程函数调用当作真实停机重启。
PROCESS_RUN_ID = uuid.uuid4().hex


def create(instrument, fast, slow):
    module = import_module("demos.01_main_ema.strategy")
    checkpoint = import_module("demos.01_main_ema.checkpoint")
    return checkpoint.EmaCheckpoint(module.MainEmaStrategy("im-restart-readonly", None,
        module.MainEmaConfig("IM", "CFFEX", fast, slow)), instrument)


def read_saved(path):
    artifact = json.loads(Path(path).read_text())
    if (artifact.get("purpose") != "readonly_restart_probe" or artifact.get("version") != 1
            or artifact.get("sha256") != digest(artifact["payload"])):
        raise HistoryError("重启指标证据类型或摘要不一致")
    return artifact["payload"]


def run_stage(service, instrument, day, path, *, stage, minutes=240, fast=3, slow=5):
    if stage not in {"prepare", "resume"} or not 0 < fast < slow <= minutes - 2 <= 238:
        raise ValueError("须指定prepare/resume且预留两根后仍有slow所需分钟")
    if not instrument.startswith("IM") or not instrument.endswith(".CFFEX"):
        raise ValueError("本探针限定固定IM合约")
    path = Path(path)
    window = ImDayWindow(day)
    stamps = window.last_minutes(window.end_ns, minutes)
    if stage == "prepare":
        if path.exists():
            raise FileExistsError("指标证据已存在，不覆盖")
        history = service.load(instrument, stamps[:-2], phase="recovery")
        if len(history.bars) < slow or history.bars[-1].ts_event != stamps[-3]:
            raise HistoryUnavailable("保存断点分钟缺失或有效分钟不足，不制造断点")
        checkpoint = create(instrument, fast, slow)
        for bar in history.bars:
            if bar.adjusted_close is None:
                raise HistoryError("历史缺少复权价格")
        for bar in history.bars:
            checkpoint.warm_closed(bar.ts_event, bar.adjusted_close)
        payload = {"instrument": instrument, "trading_day": str(day), "minutes": minutes,
            "process_run_id": PROCESS_RUN_ID, "ema": checkpoint.snapshot_state(),
            "history": service.last_audit}
        write_report(path, {"purpose": "readonly_restart_probe", "version": 1,
            "payload": payload, "sha256": digest(payload)})
        return {"status": "prepared", "stage": stage, "indicator_file": str(path),
            "process_run_id": PROCESS_RUN_ID, "saved_ema": payload["ema"],
            "history": payload["history"], "orders_submitted": 0, "targets_captured": 0,
            "ctp_connected": False, "next_step": "exit_then_start_resume_command"}

    payload = read_saved(path)
    if (payload["instrument"] != instrument or payload["trading_day"] != str(day)
            or payload["minutes"] != minutes or payload["ema"]["last_processed_ns"] != stamps[-3]):
        raise HistoryError("保存文件与本次日期、窗口或合约不一致")
    checkpoint = create(instrument, fast, slow)
    checkpoint.restore_state(payload["ema"])
    restored = checkpoint.snapshot_state()
    if restored != payload["ema"]:
        raise RuntimeError("新启动恢复状态与保存状态不一致")
    # 只补齐断点之后、作为新事件的末根之前的分钟；这里恰为14:58。
    gap = service.load(instrument, stamps[-2:-1], phase="recovery")
    gap_audit = service.last_audit
    if gap.missing or len(gap.bars) != 1 or gap.bars[0].adjusted_close is None:
        raise HistoryUnavailable("停机后的回补分钟不完整，不能跳过接续验证")
    checkpoint.warm_closed(gap.bars[0].ts_event, gap.bars[0].adjusted_close)
    after_backfill = checkpoint.snapshot_state()
    if after_backfill["revision"] or after_backfill["last_target"] is not None:
        raise RuntimeError("停机回补重放了目标")
    full = service.load(instrument, stamps, phase="recovery")
    full_audit = service.last_audit
    if (not full.bars or full.bars[-1].ts_event != stamps[-1]
            or any(bar.adjusted_close is None for bar in full.bars)):
        raise HistoryUnavailable("连续对照或末根真实事件缺失")
    baseline = create(instrument, fast, slow)
    for bar in full.bars[:-1]:
        baseline.warm_closed(bar.ts_event, bar.adjusted_close)
    # 不默许保存之前的源修订或数据补入改变序列。
    if baseline.snapshot_state() != after_backfill:
        raise HistoryError("重新读取的连续前缀与恢复回补不一致，核对历史修订")
    tail = full.bars[-1]
    if any(getattr(tail, key) is None for key in ("open", "high", "low", "close", "cumulative_factor")):
        raise HistoryError("末根事件须有真实OHLC和对应因子")
    targets = []
    strategy = checkpoint.strategy
    strategy._bind(SimpleNamespace(submit=targets.append))
    assignment = SimpleNamespace(instrument=lambda product, role: instrument.split(".")[0],
        factor=lambda product, role: tail.cumulative_factor)
    strategy.data_hub = SimpleNamespace(snapshot=lambda stamp: assignment)
    def event(bar):
        return make_bar(InstrumentId.from_str(instrument), bar.open, bar.high, bar.low, bar.close, 0, bar.ts_event)
    strategy.on_bar("fixed-restart-probe", event(tail))
    baseline.warm_closed(tail.ts_event, tail.adjusted_close)
    after = checkpoint.snapshot_state()
    expected = baseline.snapshot_state()
    if (any(after[key] != expected[key] for key in ("fast", "slow", "bars_used", "last_processed_ns"))
            or len(targets) != 1 or after["revision"] != 1 or targets[0].ts_event != tail.ts_event):
        raise RuntimeError("重启后新分钟与连续递推/目标计数不一致")
    strategy.on_bar("fixed-restart-probe", event(tail))
    if checkpoint.snapshot_state() != after or len(targets) != 1:
        raise RuntimeError("新分钟重复推进")
    return {"status": "passed", "stage": stage, "process_run_id": PROCESS_RUN_ID,
        "saved_process_run_id": payload["process_run_id"],
        "separate_process_observed": PROCESS_RUN_ID != payload["process_run_id"],
        "indicator_file": str(path), "restored_ema": restored, "after_backfill": after_backfill,
        "after_event": after, "backfill_history": gap_audit, "full_history": full_audit,
        "coverage_complete": not bool(full.missing), "continuous_ema_matches": True,
        "historical_targets_replayed": 0, "duplicate_new_event_ignored": True,
        "targets_captured": len(targets), "captured_target": str(strategy.last_target),
        "target_sink": "in_memory_probe", "orders_submitted": 0, "ctp_connected": False,
        "trading_state_written": False, "restart_scope": "staged_indicator_only"}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("prepare", "resume"), required=True)
    parser.add_argument("--indicator-file", type=Path, required=True)
    extra, remaining = parser.parse_known_args(argv)
    args = parse_warmup_args(remaining)
    if args.report and args.report.resolve() == extra.indicator_file.resolve():
        raise ValueError("报告与指标文件须使用不同路径")
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    module = import_module("demos.01_main_ema.history")
    service = module.build_history(SimpleNamespace(history_config=args.config, history_db="dolphindb",
        history_file=None, history_missing_policy=args.history_missing_policy, reference_timeout=args.timeout))
    if service.price_adjuster is None:
        raise ValueError("须配置显式factor_source_days")
    result = run_stage(service, args.instrument, args.day, extra.indicator_file,
        stage=extra.stage, minutes=args.minutes, fast=args.fast, slow=args.slow)
    if args.report:
        write_report(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
