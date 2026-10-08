"""主力EMA SimNow接入的集中离线验收；所有子进程均不联网、不读真实凭据。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "td": ("tests.run_p4_ctp_td_transport",),
    "fields": ("tests.run_p4_ctp_native_order",),
    "driver": ("tests.run_p4_ctp_driver",),
    "autosave": ("tests.run_p4_ctp_autosave",),
    "recovery": ("tests.run_p4_ctp_recovery",),
    "controlled": ("tests.run_p3_controlled_live", "--stage", "all"),
    "health": ("tests.run_market_health_gate", "--stage", "all"),
    "ctp": ("tests.run_main_ema_simnow", "--stage", "ctp"),
    "entry": ("tests.run_main_ema_live",),
}
SOURCES = (
    "bomber/__init__.py", "bomber/framework/trader/runner.py",
    "bomber/framework/trader/execution/ctp/limit_planner.py",
    "bomber/framework/trader/execution/ctp/__init__.py",
    "bomber/framework/trader/execution/ctp/native_driver.py",
    "bomber/framework/trader/execution/ctp/td_transport.py",
    "bomber/framework/trader/persistence.py", "demos/01_main_ema/strategy.py",
    "demos/01_main_ema/live_runner.py", "demos/01_main_ema/live_references.py",
    "demos/01_main_ema/run_live.py", "tests/run_main_ema_live.py",
    "tests/run_main_ema_simnow.py", "tests/run_main_ema_acceptance.py",
    "tests/run_p4_ctp_autosave.py", "tests/run_p4_ctp_recovery.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(CASES), action="append", help="只复测指定失败模块，可重复")
    parser.add_argument("--timeout", type=float, default=180, help="每模块超时秒数")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "tests/results")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("超时须大于零")
    selected = {key: value for key, value in CASES.items() if not args.only or key in args.only}
    if args.list:
        for key, command in selected.items():
            print(key, sys.executable, "-u -m", *command)
        return 0
    report_dir = args.report_dir / f"main-ema-acceptance-{time.time_ns()}"
    report_dir.mkdir(parents=True, exist_ok=False)
    result = {"python": sys.executable, "project": str(PROJECT_ROOT), "status": "running",
              "sources": {p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
              "modules": []}
    print(f"无网络集中验收 Python={sys.executable} 项目={PROJECT_ROOT} 结果={report_dir}", flush=True)
    for key, command in selected.items():
        started = time.monotonic()
        print(f"\n验证模块：{key}", flush=True)
        try:
            completed = subprocess.run([sys.executable, "-u", "-m", *command],
                cwd=PROJECT_ROOT, timeout=args.timeout, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True)
            code, output = completed.returncode, completed.stdout
        except subprocess.TimeoutExpired as error:
            raw = error.stdout or ""
            output = (raw.decode(errors="replace") if isinstance(raw, bytes) else raw) + "\n模块超时，未通过\n"
            code = 124
        (report_dir / f"{key}.log").write_text(output, encoding="utf-8")
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        row = {"module": key, "exit_code": code, "seconds": round(time.monotonic() - started, 3)}
        result["modules"].append(row)
        result["status"] = "failed" if code else "running"
        (report_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"模块结果: {row}", flush=True)
        if code:
            print(f"停止。同步修复后仅复测：python -u -m tests.run_main_ema_acceptance --only {key}", flush=True)
            return 1
    result["status"] = "passed"
    (report_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("全部选定离线模块通过；仍须本次柜台只读、在线Recording及受控订单验收。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
