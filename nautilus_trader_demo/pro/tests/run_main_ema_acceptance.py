"""主力EMA SimNow接入的集中离线验收；所有子进程均不联网、不读真实凭据。"""

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "history-adjustment": ("tests.test_history_adjustment",),
    "im-runtime": ("tests.test_main_ema_im_runtime",),
    "history-provider": ("tests.test_history_provider",),
    "continuous": ("tests.test_main_ema_continuous",),
    "gap03": ("tests.test_gap03",),
    "joint": ("tests.test_joint_live",),
    "attribution": ("tests.test_position_attribution",),
    "dispatch": ("tests.run_execution_dispatch",),
    "state": ("tests.run_state_recovery",),
    "historical": ("tests.run_unified_historical_runtime",),
    "assembly": ("tests.test_strategy_assembly",),
    "runtime": ("tests.run_runtime",),
    "references": ("tests.test_reference_sources",),
    "reference-regression": ("unittest", "tests.test_dataprep_external_factors",
        "tests.test_dataprep_role_scene", "tests.test_basic_alignment", "tests.test_datahub_option_basic"),
    "role-research": ("tests.run_role_research",),
    "role-signal": ("tests.test_demo_role_cross_three_roles",),
    "td": ("tests.run_p4_ctp_td_transport",),
    "close-existing": ("tests.test_ctp_close_existing",),
    "fields": ("tests.run_p4_ctp_native_order",),
    "driver": ("tests.run_p4_ctp_driver",),
    "autosave": ("tests.run_p4_ctp_autosave",),
    "recovery": ("tests.run_p4_ctp_recovery",),
    "controlled": ("tests.run_p3_controlled_live", "--stage", "all"),
    "health": ("tests.run_market_health_gate", "--stage", "all"),
    "routing": ("tests.run_main_ema_simnow", "--stage", "routing"),
    "ctp": ("tests.run_main_ema_simnow", "--stage", "ctp"),
    "entry": ("tests.run_main_ema_live",),
}
SOURCES = (
    "bomber/framework/dataprep/history/__init__.py", "bomber/framework/dataprep/history/base.py",
    "bomber/framework/dataprep/history/file.py", "bomber/framework/dataprep/history/dolphindb.py",
    "bomber/framework/dataprep/history/factory.py", "bomber/framework/dataprep/history/service.py",
    "demos/01_main_ema/history.py", "tests/test_history_provider.py",
    "demos/01_main_ema/history_im_raw.json", "scripts/integration/history/__init__.py",
    "scripts/integration/history/readonly.py",
    "bomber/framework/dataprep/history/factors.py", "bomber/framework/dataprep/history/windows.py",
    "scripts/integration/history/warmup.py", "scripts/integration/history/resume.py",
    "scripts/integration/history/restart.py",
    "tests/test_main_ema_im_runtime.py", "demos/01_main_ema/cffex_im_20260921_22.json",
    "demos/01_main_ema/history_im_20260922.json",
    "tests/test_history_adjustment.py",
    "demos/01_main_ema/continuous.py", "demos/01_main_ema/checkpoint.py",
    "demos/01_main_ema/check_recovery.py",
    "demos/01_main_ema/shfe_cu_2026.json", "tests/test_main_ema_continuous.py",
    "bomber/framework/trader/runtime/trading_sessions.py",
    "bomber/framework/dataprep/sources/policy.py",
    "bomber/framework/dataprep/sources/artifacts.py",
    "bomber/framework/dataprep/reference_freshness.py",
    "scripts/integration/reference_data/export.py",
    "scripts/integration/reference_data/verify_export.py",
    "scripts/integration/reference_data/readonly.py", "tests/test_gap03.py",
    "bomber/framework/trader/runtime/ownership.py",
    "bomber/framework/trader/runtime/joint.py",
    "bomber/framework/trader/execution/live/binance_account.py",
    "bomber/framework/trader/execution/live/binance_orders.py",
    "bomber/framework/trader/execution/live/binance_joint.py",
    "bomber/framework/trader/execution/live/driver.py",
    "scripts/integration/ctp_binance/run.py", "tests/test_joint_live.py",
    "scripts/integration/ctp_binance/__init__.py",
    "tests/run_live_backend.py",
    "bomber/framework/trader/template.py",
    "bomber/framework/trader/execution/attribution.py",
    "bomber/framework/trader/portfolio.py",
    "bomber/framework/trader/execution/live/client.py",
    "bomber/framework/trader/execution/live/controlled.py",
    "bomber/framework/trader/execution/simulation/client.py",
    "tests/test_position_attribution.py", "tests/run_execution_dispatch.py",
    "tests/run_p3_controlled_live.py",
    "tests/run_state_recovery.py",
    "tests/run_unified_historical_runtime.py",
    "bomber/framework/dataprep/sources/__init__.py", "bomber/framework/dataprep/sources/base.py",
    "bomber/framework/dataprep/sources/dolphindb.py", "bomber/framework/dataprep/sources/factory.py",
    "bomber/framework/dataprep/sources/file.py", "bomber/framework/dataprep/sources/normalize.py",
    "bomber/framework/dataprep/live_references.py", "bomber/framework/dataprep/factors.py",
    "bomber/framework/dataprep/references.py", "bomber/framework/dataprep/metadata.py",
    "tests/test_basic_alignment.py", "tests/test_reference_sources.py",
    "bomber/framework/datahub/role_prices.py", "tests/test_datahub_option_basic.py",
    "tests/run_role_research.py", "tests/test_demo_role_cross_three_roles.py",
    "bomber/__init__.py", "bomber/framework/trader/runner.py",
    "bomber/framework/market/stream/aggregation.py", "bomber/framework/market/stream/__init__.py",
    "bomber/framework/trader/execution/ctp/limit_planner.py",
    "bomber/framework/trader/execution/ctp/__init__.py",
    "bomber/framework/trader/execution/ctp/native_driver.py",
    "bomber/framework/trader/execution/ctp/td_transport.py",
    "bomber/framework/market/native/ctp/binding/src/bomber_ctp_td.cpp",
    "scripts/integration/ctp/close_existing.py", "tests/test_ctp_close_existing.py",
    "bomber/framework/trader/persistence.py", "demos/01_main_ema/strategy.py",
    "bomber/framework/trader/live_roles.py", "bomber/framework/dataprep/live_role.py",
    "bomber/framework/trader/runtime/ctp_positions.py",
    "bomber/framework/trader/assembly.py", "bomber/framework/trader/execution/builders.py",
    "bomber/framework/trader/runtime/managed.py", "bomber/framework/trader/runtime/ctp.py",
    "bomber/framework/trader/runtime/reports.py", "demos/01_main_ema/run_backtest.py",
    "tests/test_strategy_assembly.py", "tests/run_runtime.py",
    "demos/01_main_ema/run_live.py", "tests/run_main_ema_live.py",
    "tests/run_main_ema_simnow.py", "tests/run_main_ema_acceptance.py",
    "tests/run_p4_ctp_autosave.py", "tests/run_p4_ctp_recovery.py",
)


def check_reference_deployment():
    checks = (
        ("bomber.framework.trader.execution.ctp.td_transport", ("CtpTdApiTransport.query_position_details",)),
        ("scripts.integration.ctp.close_existing", ("inspect_account", "run_session", "parse_args")),
        ("tests.test_ctp_close_existing", (
            "CloseExistingTests.test_default_precheck_has_no_orders_and_closes_api",
            "CloseExistingTests.test_today_yesterday_and_long_direction_close_exactly_one_without_opening")),
        ("bomber.framework.market.stream.aggregation", ("ReceiveTimeTradeTickBarFeed",)),
        ("bomber.framework.market.stream", ("ReceiveTimeTradeTickBarFeed",)),
        ("bomber.framework.datahub.role_prices", ("RolePriceStore",)),
        ("bomber.framework.dataprep.metadata", ("parse_future_terms", "future_spec_from_basic")),
        ("bomber.framework.dataprep.live_references", ("LiveFuturesReferences",)),
        ("bomber.framework.dataprep.reference_freshness", ("ReferenceFreshnessPolicy",)),
        ("bomber.framework.dataprep.sources.factory", ("ReferenceSourceFactory.create_for",)),
        ("bomber.framework.dataprep.sources.artifacts", ("ExportedReferenceSource", "export_reference_bundle")),
        ("tests.test_gap03", ("Gap03Tests.test_purpose_matrix_rejects_before_constructing_or_connecting",
            "Gap03Tests.test_roundtrip_preserves_types_scope_versions_and_online_snapshot",)),
        ("bomber.framework.dataprep.live_role", ("FileRoleReferences", "SourceRoleReferences")),
        ("bomber.framework.trader.live_roles", ("FixedRoleLiveRunner", "SessionRoleLiveRunner")),
        ("bomber.framework.trader.assembly", ("StrategyBindings", "RoleGuard", "assemble_strategy")),
        ("bomber.framework.trader.execution.builders", ("ExecutionComponents", "build_simulation_execution",
            "build_ctp_execution", "attach_ctp_persistence")),
        ("bomber.framework.trader.runtime.managed", ("ManagedLiveRuntime", "LiveSessionControllerPort")),
        ("bomber.framework.trader.runtime.ctp", ("CtpSessionLifecycle",)),
        ("bomber.framework.trader.runtime.ctp_positions", ("inspect_adopted_position", "AdoptedCtpPosition")),
        ("bomber.framework.trader.runtime.reports", ("LiveRunReport",)),
        ("demos.01_main_ema.run_backtest", ("assemble",)),
        ("demos.01_main_ema.run_live", ("SourceRoleReferences", "SessionRoleLiveRunner", "build_runtime")),
        ("tests.test_strategy_assembly", (
            "AssemblyTests.test_shared_assembly_preserves_next_event_fill_and_position_accounting",
            "AssemblyTests.test_current_backtest_entry_assembles_and_runs_current_strategy",
            "ManagedRuntimeTests.test_shutdown_failure_cannot_report_passed_and_still_releases")),
        ("tests.test_basic_alignment", (
            "BasicAlignmentTests.test_static_entry_still_rejects_version_fields_and_terms_share_validation",)),
        ("tests.test_datahub_option_basic", (
            "OptionBasicTests.test_same_event_cannot_belong_to_two_trading_days",
            "OptionBasicTests.test_repeated_unsorted_closes_preserve_asof_prices_and_roll_factor",)),
        ("tests.run_role_research", ("test_missing_roll_factor_can_be_skipped_for_probe",)),
        ("tests.test_reference_sources", (
            "ReferenceSourceTests.test_public_role_adapters_support_secondary_file_and_source",
            "ReferenceSourceTests.test_database_adapter_defaults_to_role_source_day",
            "ReferenceSourceTests.test_source_day_cross_date_gap_uses_role_day_not_calendar_yesterday",)),
        ("tests.run_main_ema_live", (
            "LiveTests.test_adopted_short_closes_before_long_open_and_duplicate_fill_is_ignored",
            "LiveTests.test_adoption_rejects_wrong_gross_missing_cost_and_active_orders",
            "LiveTests.test_runtime_holds_shared_and_legacy_account_locks_before_td_start",
            "LiveTests.test_database_input_failure_closes_source_and_preflight_td",
            "LiveTests.test_runtime_construction_does_not_connect_or_open_database",
            "LiveTests.test_public_fixed_role_runner_supports_non_ema_secondary_strategy",
            "LiveTests.test_complete_replay_database_recording_entry",
            "LiveTests.test_complete_replay_database_simnow_entry",
            "LiveTests.test_replay_receive_bars_preserve_raw_ticks_and_record_current_targets",)),
    )
    for name, required_exports in checks:
        module = import_module(name)
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"部署核对：{name} 路径={path} SHA256={digest}", flush=True)
        missing = []
        for item in required_exports:
            value = module
            for part in item.split("."):
                value = getattr(value, part, None)
                if value is None:
                    missing.append(item)
                    break
        if missing:
            raise RuntimeError(f"部署版本不完整：{path} 缺少 {missing}，请同步本次运行代码与测试文件")


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
    if any(key in selected for key in ("close-existing", "assembly", "runtime", "references", "reference-regression", "role-research", "role-signal", "entry")):
        try:
            check_reference_deployment()
        except Exception as error:
            result.update(status="failed", deployment_error=str(error))
            (report_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False,
                indent=2) + "\n", encoding="utf-8")
            print(f"部署核对失败：{error}。尚未开始测试。", flush=True)
            return 2
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
