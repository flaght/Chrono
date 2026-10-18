"""指定IM合约/日期的复权分钟及EMA预热只读核验；不连接CTP、不写交易状态。"""
import argparse
from datetime import date, datetime, timezone
from decimal import Decimal
from importlib import import_module
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from bomber.framework.dataprep.history import ImDayWindow
from .readonly import write_report


def probe(service, instrument, trading_day, *, minutes=240, fast=3, slow=5):
    if not 0 < fast < slow <= minutes <= 240:
        raise ValueError("须满足0<fast<slow<=minutes<=240")
    if not instrument.startswith("IM") or not instrument.endswith(".CFFEX"):
        raise ValueError("本只读预热入口限定固定IM合约")
    strategy_module = import_module("demos.01_main_ema.strategy")
    checkpoint_module = import_module("demos.01_main_ema.checkpoint")
    warm_history = import_module("demos.01_main_ema.history").warm_history
    strategy = strategy_module.MainEmaStrategy("im-history-readonly", None,
        strategy_module.MainEmaConfig("IM", "CFFEX", fast, slow, Decimal(1)))
    window = ImDayWindow(trading_day)
    checkpoint = checkpoint_module.EmaCheckpoint(strategy, instrument)
    runner = SimpleNamespace(ema_checkpoint=checkpoint, calendar=window, history_service=service,
        history_minutes=minutes, _submit_lock=RLock(), _clients={"readonly": SimpleNamespace(_submit_lock=RLock())})
    result = warm_history(runner, window.end_ns, phase="recovery")
    if (strategy._revision or strategy.last_target is not None or strategy.fills_received
            or strategy.order_updates_received or (result.missing and not result.allowed_missing)
            or strategy.bars_used != len(result.bars)
            or not strategy.slow.initialized):
        raise RuntimeError("只读预热覆盖/指标/无历史交易要求未满足")
    first, last = result.bars[0], result.bars[-1]
    def sample(bar):
        return {"ts_event": bar.ts_event, "raw_close": str(bar.close),
            "adjusted_close": str(bar.adjusted_close)}
    missing_minutes = [datetime.fromtimestamp(stamp // 10**9,
        timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).isoformat() for stamp in result.missing]
    return {"status": "allowed_missing" if result.missing else "passed",
        "coverage_complete": not bool(result.missing), "missing_minutes": missing_minutes,
        "purpose": "fixed_im_day_adjusted_ema_warmup_readonly",
        "instrument": instrument, "trading_day": str(trading_day), "history": service.last_audit,
        "first": sample(first), "last": sample(last), "ema": checkpoint.snapshot_state(),
        "ema_warmed": True, "orders_submitted": 0, "targets_published": 0,
        "state_written": False, "ctp_connected": False}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--trading-day", required=True, help="已核验交易日YYYY-MM-DD")
    parser.add_argument("--minutes", type=int, default=240)
    parser.add_argument("--fast", type=int, default=3)
    parser.add_argument("--slow", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--history-missing-policy", choices=("fail", "allow"), default="fail")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not args.connect or not 1 <= args.timeout <= 120 or not 0 < args.fast < args.slow <= args.minutes <= 240:
        parser.error("须显式connect，超时1至120秒，且0<fast<slow<=minutes<=240")
    if args.report and args.report.exists():
        parser.error("报告已存在，保留旧证据")
    day = date.fromisoformat(args.trading_day)
    window = ImDayWindow(day)
    if window.end_ns > int(datetime.now(timezone.utc).timestamp()) * 10**9:
        parser.error("须使用完整结束的指定交易日")
    args.day = day
    return args


def main(argv=None):
    args = parse_args(argv)
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    history_module = import_module("demos.01_main_ema.history")
    service = history_module.build_history(SimpleNamespace(history_config=args.config, history_db="dolphindb",
        history_file=None, history_missing_policy=args.history_missing_policy, reference_timeout=args.timeout))
    if service.price_adjuster is None:
        raise ValueError("配置须提供factor_source_days，不能用当前因子或默认1替代")
    result = probe(service, args.instrument, args.day, minutes=args.minutes, fast=args.fast, slow=args.slow)
    if args.report:
        write_report(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
