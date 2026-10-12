"""Run actual 08 Recording and review its exact report; never authorizes TD orders."""
import argparse
from datetime import datetime, timedelta
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trading-day", required=True)
    parser.add_argument("--option-slot", default="13:58")
    parser.add_argument("--seconds", type=int, default=4200)
    parser.add_argument("--ready-timeout", type=int, default=180)
    parser.add_argument("--report-dir", type=Path, default=Path("tests/results/gap08-recording"))
    parser.add_argument("--summary", type=Path, help="仅复核已有summary.json，不连接")
    parser.add_argument("--allow-no-target", action="store_true", help="只验决策评价，不声称非零目标链通过")
    args = parser.parse_args(argv)
    worker_exit_code = None
    root = Path(__file__).resolve().parents[3]
    review = import_module("demos.08_option_vega.acceptance").review_recording
    if args.summary:
        path = args.summary.resolve()
        destination = path.parent / "recording-acceptance.json"
    else:
        try:
            day = datetime.strptime(args.trading_day, "%Y%m%d").date()
            hour, minute = map(int, args.option_slot.split(":"))
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            decision = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now.date() != day or now + timedelta(seconds=args.ready_timeout+65) >= decision:
                raise ValueError("请在目标交易日选约时点前至少ready-timeout+65秒启动")
            if args.seconds < 60 or args.ready_timeout < 1 or now+timedelta(seconds=args.seconds) <= decision+timedelta(seconds=10):
                raise ValueError("运行时长须覆盖选约分钟及其等待时间")
        except ValueError as error:
            parser.error(str(error))
        output = args.report_dir.resolve() / f"acceptance-{time.time_ns()}"
        output.mkdir(parents=True)
        destination = output / "recording-acceptance.json"
        command = [sys.executable, "-u", "-m", "demos.08_option_vega.run_live",
            "--connect", "--mode", "recording", "--market-source", "ctp", "--ctp-price-type", "last",
            "--simnow-environment", "realtime", "--expected-trading-day", args.trading_day,
            "--valuation-day", args.trading_day, "--calendar-id", "china.cffex",
            "--reference-source", "dolphindb", "--option-product", "MO", "--future-product", "IM",
            "--index-code", "000852", "--index-source", "dolphindb-dfs",
            "--index-feature-config", str(root/"bomber/framework/dataprep/features/configs/features.json"),
            "--index-dataset", "index.minute", "--index-schema-version", "1",
            "--index-database", "dfs://min_bar", "--index-table", "index_1min",
            "--index-bar-timestamp", "end", "--index-availability", "observed",
            "--index-price-precision", "4", "--index-size-precision", "4",
            "--index-poll-seconds", ".2", "--index-query-timeout", "2", "--frame-wait-seconds", "5",
            "--option-slot", args.option_slot, "--ready-timeout", str(args.ready_timeout), "--seconds", str(args.seconds),
            "--max-option-lots", "1", "--max-total-option-lots", "2", "--max-future-lots", "1",
            "--max-notional", "2000000", "--report-dir", str(output/"runtime")]
        print("启动真实Recording；不连接TD、不报单", flush=True)
        try:
            process = subprocess.run(command, cwd=root, timeout=args.seconds+args.ready_timeout+180)
            worker_exit_code = process.returncode
            paths = list((output/"runtime").glob("*/summary.json"))
            if len(paths) != 1:
                raise ValueError(f"运行退出码={process.returncode}，未取得唯一summary.json")
            path = paths[0]
        except (subprocess.TimeoutExpired, ValueError) as error:
            destination.write_text(json.dumps({"status": "failed", "errors": [str(error)]}, ensure_ascii=False, indent=2))
            print(f"验收失败: {error}\n验收报告: {destination}", flush=True)
            return 1
    try:
        result = review(json.loads(path.read_text()), trading_day=args.trading_day,
            option_slot=args.option_slot, require_target=not args.allow_no_target)
    except (ValueError, KeyError, TypeError) as error:
        result = {"status": "failed", "errors": [f"报告内容不完整或无效: {error}"]}
    result["runtime_summary"] = str(path)
    result["worker_exit_code"] = worker_exit_code
    if worker_exit_code not in (None, 0):
        result["status"] = "failed"
        result.setdefault("errors", []).append(f"Recording进程退出码={worker_exit_code}")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(f"Recording验收: {result}\n验收报告: {destination}", flush=True)
    return int(result["status"] != "passed")


if __name__ == "__main__":
    sys.exit(main())
