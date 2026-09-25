#!/usr/bin/env python3
"""Run the project regression scripts that need no live market connection."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Each script gets its own process because the trading engine has process-wide
# state, including a logger which cannot always be initialized twice.
OFFLINE_CASES: tuple[tuple[str, ...], ...] = (
    ("tests/run_datahub.py",),
    ("tests/run_dynamic_routes.py",),
    ("tests/run_execution_backend.py",),
    ("tests/run_execution_dispatch.py",),
    ("tests/run_execution_events.py",),
    ("tests/run_market_health.py",),
    ("tests/run_market_health_gate.py",),
    ("tests/run_market_stream_adapter.py",),
    ("tests/run_order_state.py",),
    ("tests/run_p3_account_reader.py",),
    ("tests/run_p3_active_orders.py",),
    ("tests/run_p3_controlled_live.py",),
    ("tests/run_p4_binance_orders.py",),
    ("tests/run_p4_ctp_autosave.py",),
    ("tests/run_p4_ctp_driver.py",),
    ("tests/run_p4_ctp_native_order.py",),
    ("tests/run_p4_ctp_recovery.py",),
    ("tests/run_p4_ctp_td_transport.py",),
    ("tests/run_portfolio.py",),
    ("tests/run_reconciliation.py",),
    ("tests/run_risk_manager.py",),
    ("tests/run_role_cross.py",),
    ("tests/run_role_research.py",),
    ("tests/run_runtime.py",),
    ("tests/run_scheduled_targets.py",),
    ("tests/run_simulation_backend.py",),
    ("tests/run_single_ema_online.py", "--stage", "offline"),
    ("tests/run_state_recovery.py",),
    ("tests/run_unified_historical_runtime.py", "--stage", "all"),
    ("tests/run_ctp_high_fidelity.py",),
    ("tests/run_ctp_simnow_order_probe.py",),
    ("tests/run_ctp_ema_staging.py",),
    ("tests/run_live_backend.py", "--stage", "all"),
    ("tests/run_formal_strategy.py", "--stage", "5"),
    ("tests/run_black_sector.py",),
    ("tests/strategies/cross_section/run_test.py",),
    ("tests/strategies/single_ema/run_test.py",),
)

# These require local historical files, but do not connect to an exchange.
DATA_CASES: tuple[tuple[str, ...], ...] = (
    ("tests/run_ctp_simulation.py", "--stage", "2"),
    ("tests/run_binance_simulation.py",),
    ("tests/run_formal_strategy.py", "--stage", "6"),
    ("tests/run_formal_strategy.py", "--stage", "7"),
)

SKIPPED_LIVE = (
    "tests/run_ctp_live.py",
    "tests/run_binance_live.py",
    "tests/run_dolphin_live.py",
    "tests/run_strategy.py (default main includes a live Binance stage)",
    "tests/run_single_ema_online.py --stage 4/5",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-data", action="store_true", help="also run historical-file cases")
    parser.add_argument("--list", action="store_true", help="show selected and skipped cases")
    parser.add_argument("--verbose", action="store_true", help="print full output for passing cases")
    parser.add_argument(
        "--only", action="append", metavar="NAME",
        help="run only a named script, such as run_role_cross; may be repeated",
    )
    parser.add_argument("--timeout", type=int, default=300, help="seconds per script")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    cases = OFFLINE_CASES + (DATA_CASES if args.with_data else ())
    if args.only:
        selected = set(args.only)
        cases = tuple(
            case for case in cases
            if case[0] in selected or Path(case[0]).stem in selected
        )
        if not cases:
            parser.error("--only did not match any selected test script")
    if args.list:
        for case in cases:
            print("RUN ", " ".join(case))
        for name in SKIPPED_LIVE:
            print("SKIP", name, "(live connection)")
        if not args.with_data:
            for case in DATA_CASES:
                print("SKIP", " ".join(case), "(use --with-data)")
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(PROJECT_ROOT), env.get("PYTHONPATH"))))
    prerequisite = subprocess.run(
        [sys.executable, "-c", "import trader"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
    )
    if prerequisite.returncode:
        print("无法导入 trader；请先安装项目依赖：", file=sys.stderr)
        print(prerequisite.stderr, file=sys.stderr)
        return 1

    failed: list[str] = []
    started = time.monotonic()
    for index, case in enumerate(cases, 1):
        label = " ".join(case)
        print(f"[{index}/{len(cases)}] {label}", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, *case], cwd=PROJECT_ROOT, env=env,
                capture_output=True, text=True, timeout=args.timeout,
            )
        except subprocess.TimeoutExpired as error:
            failed.append(label)
            print(f"  FAIL: 超过 {args.timeout} 秒；请检查该测试", flush=True)
            if error.stdout:
                print(error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout)
            if error.stderr:
                print(error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr)
            continue
        if result.returncode:
            failed.append(label)
            print(f"  FAIL: exit={result.returncode}", flush=True)
            print(result.stdout[-8000:])
            print(result.stderr[-8000:], file=sys.stderr)
        else:
            print("  PASS", flush=True)
            if args.verbose:
                print(result.stdout)
                print(result.stderr, file=sys.stderr)

    print(f"结果：{len(cases) - len(failed)}/{len(cases)} 通过；耗时 {time.monotonic() - started:.1f} 秒")
    print("跳过实时连接：" + ", ".join(SKIPPED_LIVE))
    if failed:
        print("失败项：" + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
