"""Linux在线分阶段验收：只读资料、Recording、Model、受控SimNow。"""

import argparse
from datetime import datetime
from decimal import Decimal
from importlib import import_module
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def review(summary, mode, *, expected, lookback):
    errors = []
    if summary.get("status") != "passed" or summary.get("mode") != mode:
        errors.append("运行报告未通过或模式不符")
    if summary.get("cleanup_errors"):
        errors.append("停机清理失败")
    if summary.get("fixed_instruments") != expected:
        errors.append("双腿身份与只读预检不符")
    if summary.get("complete_frames", 0) < lookback + 1:
        errors.append("同步帧不足")
    if summary.get("submissions", 0) < 1:
        errors.append("没有实际策略目标，尚不能进入下一阶段")
    targets = summary.get("last_targets", {})
    if set(targets) != set(expected.values()):
        errors.append("最新目标未完整覆盖双腿")
    if mode != "recording":
        if summary.get("fills_received", 0) < 2:
            errors.append("归属成交回调不足")
        positions = summary.get("strategy_positions", {})
        if set(positions) != set(targets) or any(
                Decimal(str(positions.get(key))) != Decimal(str(value)) for key, value in targets.items()):
            errors.append("归属仓位尚未达到最新双腿目标")
        for field, label in (("working_quantities", "在途量"), ("unassigned_positions", "未归属仓位")):
            values = summary.get(field, {})
            if set(values) != set(expected.values()) or any(Decimal(str(v)) != 0 for v in values.values()):
                errors.append(f"{label}未确认全部归零")
    if mode == "simnow" and summary.get("final_active_orders") != 0:
        errors.append("停机未确认柜台无活动订单")
    return {"status": "failed" if errors else "passed", "mode": mode, "errors": errors,
        "runtime_failure": summary.get("failure"), "session_readiness": summary.get("session_readiness"),
        "complete_frames": summary.get("complete_frames"),
        "per_instrument_bar_count": summary.get("per_instrument_bar_count")}


def prepare(args):
    # 只有显式--connect后才导入原生模块并联网；构造和--help只依赖标准库。
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from scripts.integration.ctp.td_readonly import build_readonly_driver
    from bomber.framework.dataprep.live_references import LiveFuturesReferences
    from bomber.framework.dataprep.sources import (
        ReferenceSourceFactory, DataSourcePurpose, DolphinDbReferenceConfig)
    references_type = import_module("demos.03_calendar_spread.live_references").CalendarReferences
    driver = build_readonly_driver(timeout=args.query_timeout)
    try:
        driver.start(lambda report: None)
        day = driver.trading_day
        if not day or datetime.strptime(day, "%Y%m%d").strftime("%Y%m%d") != day:
            raise ValueError("柜台未提供有效交易日")
        print(f"{args.product}跨期只读预检: TD={day} environment={args.environment}; 此阶段不报单", flush=True)
    finally:
        driver.stop()
    config = DolphinDbReferenceConfig.from_env(database=args.reference_database,
        read_timeout_seconds=args.query_timeout)
    with ReferenceSourceFactory.create_for(DataSourcePurpose.LIVE_REFERENCE, "dolphindb", config) as source:
        service = LiveFuturesReferences(source, products=(args.product,), trading_day=day,
            started_ns=time.time_ns(), roles=("secondary", "far"), factor_roles=(),
            factor_date_basis="source", factor_availability="source-day-end")
        references = references_type(service, args.product, ("secondary", "far"))
        assignment = references.snapshot(time.time_ns())
        result = {"trading_day": day, "source_day": assignment.source_day.strftime("%Y%m%d"),
            "instruments": {r: str(i) for r, i in references.instrument_ids.items()},
            "reference_manifest": references.manifest}
    print(f"资料预检: {args.product} source_day={result['source_day']} legs={result['instruments']}", flush=True)
    return result


def command(args, context, mode, folder):
    seconds = args.recording_seconds if mode == "recording" else args.execution_seconds
    result = [sys.executable, "-u", "-m", "demos.03_calendar_spread.run_live", "--connect",
        "--mode", mode, "--product", args.product, "--simnow-environment", args.environment,
        "--expected-trading-day", context["trading_day"], "--expected-source-day", context["source_day"],
        "--reference-database", args.reference_database, "--leg1-role", "secondary", "--leg2-role", "far",
        "--quantity", "1", "--lookback", str(args.lookback), "--entry-z", "2", "--exit-z", ".5",
        "--rebalance-interval", "1", "--max-notional", str(args.max_notional),
        "--instrument-max-age-seconds", str(args.instrument_max_age_seconds),
        "--max-session-orders", str(args.max_session_orders), "--query-timeout", str(args.query_timeout),
        "--seconds", str(seconds), "--report-dir", str(folder / "runtime")]
    for role, instrument in context["instruments"].items():
        result.extend(("--expected-instrument", f"{role}={instrument}"))
    if args.environment == "replay":
        result.extend(("--replay-md-trading-day", args.replay_md_trading_day))
    if mode != "recording":
        result.extend(("--state-file", str(folder / "state.json"), "--require-fills"))
    if mode == "model":
        result.extend(("--strategy-position-source", "model", "--account-reconciliation", "off"))
    elif mode == "simnow":
        result.extend(("--enable-orders", "--confirm-simnow", "--account-reconciliation", "strict"))
    return result


def run_stage(args, context, mode, folder):
    folder.mkdir()
    print(f"开始阶段 {mode}: quantity=1 lookback={args.lookback}; 输出={folder}", flush=True)
    worker = command(args, context, mode, folder)
    (folder / "command.json").write_text(json.dumps(worker, ensure_ascii=False, indent=2) + "\n")
    # 独立进程组使控制器将中断转发一次，给交易入口完整撤单/保存/释放时间。
    process = subprocess.Popen(worker, cwd=ROOT, start_new_session=True)
    try:
        process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"停机仍未结束，保留运行进程PID={process.pid}，须核对柜台与状态文件") from error
        raise
    paths = tuple((folder / "runtime").glob("*/summary.json"))
    if len(paths) != 1:
        result = {"status": "failed", "mode": mode, "errors": ["未取得唯一运行summary.json"]}
    else:
        try:
            result = review(json.loads(paths[0].read_text()), mode,
                expected=context["instruments"], lookback=args.lookback)
        except (ValueError, TypeError, KeyError, ArithmeticError) as error:
            result = {"status": "failed", "mode": mode, "errors": [f"报告无效: {error}"]}
        result["runtime_summary"] = str(paths[0])
    result["worker_exit_code"] = process.returncode
    if process.returncode:
        result["status"] = "failed"
        result["errors"].append(f"运行退出码={process.returncode}")
    (folder / "acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"阶段 {mode}: {result}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--product", default="CU")
    parser.add_argument("--environment", choices=("realtime", "replay"), default="realtime")
    parser.add_argument("--replay-md-trading-day")
    parser.add_argument("--through", choices=("recording", "model", "simnow"), default="recording")
    parser.add_argument("--enable-orders", action="store_true")
    parser.add_argument("--confirm-simnow", action="store_true")
    parser.add_argument("--lookback", type=int, default=3, help="验收窗口；常规策略默认120，此处显式使用3")
    parser.add_argument("--recording-seconds", type=int, default=600)
    parser.add_argument("--execution-seconds", type=int, default=900)
    parser.add_argument("--max-notional", type=Decimal, default=Decimal(1000000))
    parser.add_argument("--max-session-orders", type=int, default=20)
    parser.add_argument("--instrument-max-age-seconds", type=float,
        help="逐合约时效(0,120]秒；默认realtime10/replay30，全流接收限制仍为10秒")
    parser.add_argument("--query-timeout", type=int, default=15)
    parser.add_argument("--reference-database", default="dfs://bomber_daily")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "tests/results/calendar-spread-online")
    args = parser.parse_args(argv)
    args.product = args.product.strip().upper()
    if args.instrument_max_age_seconds is None:
        args.instrument_max_age_seconds = 30.0 if args.environment == "replay" else 10.0
    if not args.connect:
        parser.error("须显式--connect，联网阶段仅在Linux运行")
    if not args.product.isascii() or not args.product.isalpha() or args.lookback < 2:
        parser.error("品种须为英文字母，lookback至少2")
    if args.recording_seconds < 60 or args.execution_seconds < 60 or args.query_timeout < 1 or args.max_session_orders < 1:
        parser.error("运行时长至少60秒，查询超时与订单上限须为正")
    if not args.max_notional.is_finite() or args.max_notional <= 0:
        parser.error("名义限额须为有限正数")
    if not math.isfinite(args.instrument_max_age_seconds) or not 0 < args.instrument_max_age_seconds <= 120:
        parser.error("逐合约时效须为有限(0,120]秒")
    if args.environment == "replay":
        try:
            value = args.replay_md_trading_day or ""
            if datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") != value:
                raise ValueError()
        except ValueError:
            parser.error("replay须实际原始MD日YYYYMMDD")
    elif args.replay_md_trading_day:
        parser.error("realtime不接受原始回放MD日")
    if args.through == "simnow" and not (args.enable_orders and args.confirm_simnow):
        parser.error("进入SimNow须同时--enable-orders与--confirm-simnow")
    if args.through != "simnow" and (args.enable_orders or args.confirm_simnow):
        parser.error("Recording/Model验收不接受报单授权")
    return args


def main(argv=None):
    args = parse_args(argv)
    folder = args.report_dir.resolve() / f"{args.product.lower()}-{time.time_ns()}"
    folder.mkdir(parents=True, exist_ok=False)
    stages = []
    result = {"status": "failed", "product": args.product, "environment": args.environment,
        "stages": stages, "report_directory": str(folder)}
    try:
        context = prepare(args)
        (folder / "preflight.json").write_text(json.dumps(context, ensure_ascii=False, indent=2, default=str) + "\n")
        for mode in ("recording", "model", "simnow"):
            stage = run_stage(args, context, mode, folder / mode)
            stages.append(stage)
            if stage["status"] != "passed":
                break
            if mode == args.through:
                result["status"] = "passed"
                break
    except KeyboardInterrupt:
        result["failure"] = "用户中断，检查各阶段停机报告及保留状态文件"
    except Exception as error:
        result["failure"] = str(error)
    finally:
        (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"分阶段验收 {result['status']}: {folder / 'summary.json'}", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
