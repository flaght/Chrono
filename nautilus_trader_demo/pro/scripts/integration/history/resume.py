"""固定IM真实历史的指标文件恢复和末根事件衔接；目标仅由本地探针记录。"""
import argparse
import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from bomber.framework.dataprep.history import HistoryError, ImDayWindow
from bomber.framework.market.basic.base import InstrumentId, make_bar
from .readonly import write_report
from .warmup import parse_args as parse_warmup_args


def digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_indicator(path):
    artifact = json.loads(Path(path).read_text())
    if (artifact.get("purpose") != "readonly_indicator_probe"
            or artifact.get("version") != 1
            or artifact.get("sha256") != digest(artifact["ema"])):
        raise HistoryError("只读指标文件类型或摘要不一致")
    return artifact["ema"]


def probe(service, instrument, trading_day, checkpoint_file, *, minutes=240, fast=3, slow=5):
    path = Path(checkpoint_file)
    if path.exists():
        raise FileExistsError("指标文件已存在，保留原证据并指定新路径")
    if not 0 < fast < slow < minutes <= 240:
        raise ValueError("恢复探针须满足0<fast<slow<minutes<=240")
    if not instrument.startswith("IM") or not instrument.endswith(".CFFEX"):
        raise ValueError("本探针限定固定IM合约")
    module = import_module("demos.01_main_ema.strategy")
    checkpoint_module = import_module("demos.01_main_ema.checkpoint")
    def create():
        strategy = module.MainEmaStrategy("im-resume-readonly", None,
            module.MainEmaConfig("IM", "CFFEX", fast, slow))
        return checkpoint_module.EmaCheckpoint(strategy, instrument)
    window = ImDayWindow(trading_day)
    stamps = window.last_minutes(window.end_ns, minutes)
    result = service.load(instrument, stamps, phase="recovery")
    bars = result.bars
    if (len(bars) < slow + 1 or bars[-1].ts_event != stamps[-1]
            or any(any(getattr(bar, field) is None for field in
                ("open", "high", "low", "close", "adjusted_close", "cumulative_factor")) for bar in bars)):
        raise HistoryError("须有初始化所需真实历史及末根完整原始OHLC/因子，不补造事件")
    baseline, original = create(), create()
    for bar in bars:
        baseline.warm_closed(bar.ts_event, bar.adjusted_close)
    for bar in bars[:-1]:
        original.warm_closed(bar.ts_event, bar.adjusted_close)
    saved = original.snapshot_state()
    write_report(path, {"version": 1, "purpose": "readonly_indicator_probe",
        "sha256": digest(saved), "ema": saved, "history": service.last_audit})
    restored = create()
    restored.restore_state(read_indicator(path))
    restored_state = restored.snapshot_state()
    if restored_state != saved:
        differences = {key: {"saved": saved[key], "restored": restored_state.get(key)}
            for key in saved if saved[key] != restored_state.get(key)}
        raise RuntimeError(f"文件恢复后的指标状态不一致：{differences}")
    strategy = restored.strategy
    targets = []
    strategy._bind(SimpleNamespace(submit=targets.append))
    current = bars[-1]
    # 固定合约探针显式提供已查询因子，未查询/修改生产main映射。
    assignment = SimpleNamespace(instrument=lambda product, role: instrument.split(".")[0],
        factor=lambda product, role: current.cumulative_factor)
    strategy.data_hub = SimpleNamespace(snapshot=lambda stamp: assignment)
    def event(bar):
        return make_bar(InstrumentId.from_str(instrument), bar.open, bar.high, bar.low,
            bar.close, 0, bar.ts_event)
    for old in (bars[0], bars[-2]):
        strategy.on_bar("fixed-resume-probe", event(old))
    if restored.snapshot_state() != saved or targets:
        raise RuntimeError("恢复后历史事件重复推进或重放目标")
    strategy.on_bar("fixed-resume-probe", event(current))
    after = restored.snapshot_state()
    expected = baseline.snapshot_state()
    if (after["bars_used"] != len(bars) or after["last_processed_ns"] != current.ts_event
            or not all(math.isclose(float(after[key]), float(expected[key]), rel_tol=1e-12, abs_tol=1e-9)
                for key in ("fast", "slow"))
            or len(targets) != 1 or strategy._revision != 1
            or targets[0].ts_event != current.ts_event):
        raise RuntimeError("首根新事件与连续指标或目标计数不一致")
    strategy.on_bar("fixed-resume-probe", event(current))
    if restored.snapshot_state() != after or len(targets) != 1:
        raise RuntimeError("重复新分钟产生重复指标或目标")
    if strategy.fills_received or strategy.order_updates_received:
        raise RuntimeError("只读探针出现交易回调")
    return {"status": "passed", "purpose": "fixed_im_indicator_file_resume_readonly",
        "coverage_complete": not bool(result.missing), "history": service.last_audit,
        "instrument": instrument, "trading_day": str(trading_day),
        "indicator_file": str(path), "indicator_sha256": digest(saved),
        "saved_ema": saved, "restored_ema_before_event": restored_state, "after_event": after,
        "continuous_ema": expected, "continuous_ema_matches": True,
        "historical_targets_replayed": 0, "old_events_ignored": 2,
        "new_event_ns": current.ts_event, "duplicate_new_event_ignored": True,
        "targets_captured": len(targets), "captured_target": str(strategy.last_target),
        "target_sink": "in_memory_probe", "orders_submitted": 0, "ctp_connected": False,
        "trading_state_written": False, "restart_scope": "new_instance_from_json_file"}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--indicator-file", type=Path, required=True)
    extra, remaining = parser.parse_known_args(argv)
    args = parse_warmup_args(remaining)
    if extra.indicator_file.exists() or (args.report and args.report.resolve() == extra.indicator_file.resolve()):
        raise ValueError("指标文件须使用独立的新路径")
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    history = import_module("demos.01_main_ema.history")
    service = history.build_history(SimpleNamespace(history_config=args.config, history_db="dolphindb",
        history_file=None, history_missing_policy=args.history_missing_policy, reference_timeout=args.timeout))
    if service.price_adjuster is None:
        raise ValueError("配置须显式提供factor_source_days")
    report = probe(service, args.instrument, args.day, extra.indicator_file,
        minutes=args.minutes, fast=args.fast, slow=args.slow)
    if args.report:
        write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
