"""04 SimNow集中离线验收；记录版本/日志/退出码，首个失败即停止。"""

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "cross-section": ("tests.test_cross_section_live",),
    "signal": ("tests.test_demo_cross_section_signal",),
    "entry": ("tests.run_main_ema_live",),
    "routing": ("tests.run_main_ema_simnow", "--stage", "routing"),
    "ctp": ("tests.run_main_ema_simnow", "--stage", "ctp"),
    "assembly": ("tests.test_strategy_assembly",),
    "dispatch": ("tests.run_execution_dispatch",),
    "attribution": ("tests.test_position_attribution",),
}
SOURCES = (
    "bomber/framework/market/stream/ctp/feed.py",
    "bomber/framework/market/stream/ctp/converter.py",
    "bomber/framework/trader/runner.py", "bomber/framework/trader/execution/builders.py",
    "bomber/framework/trader/execution/ctp/limit_planner.py",
    "bomber/framework/trader/runtime/ctp.py", "bomber/framework/trader/runtime/managed.py",
    "bomber/framework/trader/persistence.py", "bomber/framework/trader/assembly.py",
    "bomber/framework/dataprep/live_references.py", "bomber/framework/dataprep/live_role.py",
    "bomber/framework/trader/execution/ctp/native_driver.py",
    "bomber/framework/trader/execution/ctp/td_transport.py",
    "bomber/framework/trader/execution/attribution.py",
    "bomber/framework/trader/execution/live/client.py",
    "bomber/framework/trader/execution/live/controlled.py",
    "demos/04_cross_section/run_live.py", "demos/04_cross_section/live_runtime.py",
    "demos/04_cross_section/recovery.py",
    "demos/04_cross_section/strategy.py", "demos/04_cross_section/cross_section_signal.py",
    "demos/01_main_ema/run_live.py", "demos/01_main_ema/strategy.py",
    "tests/test_cross_section_live.py", "tests/run_main_ema_live.py",
    "tests/run_p4_ctp_td_transport.py", "tests/test_demo_cross_section_signal.py",
    "tests/run_cross_section_acceptance.py", "tests/run_main_ema_simnow.py",
    "tests/test_strategy_assembly.py", "tests/run_execution_dispatch.py",
    "tests/test_position_attribution.py",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(CASES), action="append")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "tests/results")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("timeout须为正")
    path = args.report_dir / f"cross-section-acceptance-{time.time_ns()}"
    path.mkdir(parents=True, exist_ok=False)
    report = {"python": sys.executable, "project": str(ROOT), "status": "running", "modules": [],
        "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}}
    print(f"离线验收 Python={sys.executable} project={ROOT} report={path}", flush=True)
    try:
        for name in ("bomber.framework.trader.execution.builders", "bomber.framework.trader.runner",
                     "demos.04_cross_section.run_live", "demos.04_cross_section.strategy"):
            module = import_module(name)
            actual = Path(module.__file__).resolve()
            if not actual.is_relative_to(ROOT):
                raise RuntimeError(f"模块未加载本次项目源码: {name} {actual}")
            print(f"加载路径: {name}={actual} SHA256={hashlib.sha256(actual.read_bytes()).hexdigest()}", flush=True)
        entry = import_module("demos.04_cross_section.run_live")
        runtime = import_module("demos.04_cross_section.live_runtime")
        feed = import_module("bomber.framework.market.stream.ctp.feed")
        expected_interfaces = (
            (entry, "parse_args", "04 run_live.parse_args"),
            (runtime.CrossSectionSessionRunner, "bar_is_current", "04 runner.bar_is_current"),
            (feed.CtpLiveDataFeed, "depth_observations", "CTP feed.depth_observations"),
        )
        for owner, attribute, label in expected_interfaces:
            if not hasattr(owner, attribute):
                raise RuntimeError(f"部署文件版本不一致，缺少{label}；同步PF2026100924清单全部文件并核对SHA后再测试")
    except Exception as error:
        report.update(status="failed", deployment_error=str(error))
        (path / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"部署/加载核对失败: {error}", flush=True)
        return 2
    for key, command in CASES.items():
        if args.only and key not in args.only:
            continue
        started = time.monotonic()
        try:
            completed = subprocess.run([sys.executable, "-u", "-m", *command], cwd=ROOT,
                timeout=args.timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            code, output = completed.returncode, completed.stdout
        except subprocess.TimeoutExpired as error:
            raw = error.stdout or ""
            output = (raw.decode(errors="replace") if isinstance(raw, bytes) else raw) + "\n模块超时\n"
            code = 124
        (path / f"{key}.log").write_text(output, encoding="utf-8")
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        row = {"module": key, "exit_code": code, "seconds": round(time.monotonic() - started, 3)}
        report["modules"].append(row)
        report["status"] = "failed" if code else "running"
        (path / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"模块结果: {row}", flush=True)
        if code:
            print(f"停止，仅复测失败组: python -u -m tests.run_cross_section_acceptance --only {key}", flush=True)
            return 1
    report["status"] = "passed"
    (path / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("全部选定离线模块通过；实际柜台Recording/组合成交仍须独立验收。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
