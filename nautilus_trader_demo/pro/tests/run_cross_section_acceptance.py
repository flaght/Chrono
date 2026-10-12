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
FEATURE_DEPLOYMENT_GROUPS = frozenset(("feature-datahub", "feature-query-live", "option-vega", "option-vega-live", "option-calendar"))
# 补充原生/08被测文件摘要；入口摘要由独立GAP08入口持有，不在此处自引用。
FEATURE_RUNTIME_HASHES = {
    "bomber/framework/calendars/DateUtilities/Calendar.pyx": "efcf365e9795bb1669527c3639b21d28d99627119f772ed8a219502c943543eb",
    "bomber/framework/calendars/__init__.py": "49c3010d35d57ac88dec9f7eb2e81648786318c98bc27f7a6800ac2ffba2504a",
    "bomber/framework/calendars/trading.py": "94a4eaaaf5306b82b71d5117183db1fff0cafb2031fc4f962b7d138a5b4816fa",
    "bomber/framework/dataprep/calendar.py": "ad97c156acf7dd915e9727e30110efb1e5826e41f3512051c423b7462771e7ec",
    "bomber/framework/dataprep/live_options.py": "3222662581df4c0729e2eb4a2e077c9d83938ba35b424cb49ad27481943c4f0c",
    "bomber/framework/market/stream/feature_query.py": "46efd38358805bf460624f82bc1a89b8371c1107ec50dabd5d1ad1cca498179e",
    "bomber/framework/trader/runtime/live/channels/ctp_recovery.py": "8764545ae3ba22ddb5cbd242ca3447fa5aa569a9615bbde90737c883584339e8",
    "bomber/framework/trader/runtime/live/channels/features.py": "bc3fba69836682cd6cf5d71014c2f99cb6182f6011bb80fcfa5125f667c55288",
    "bomber/framework/trader/runtime/live/readiness.py": "125a27f8b9ad3ede9809fa0865dfac90f44572f7e979f22c1e115db19883e96b",
    "demos/08_option_vega/acceptance.py": "5e557fa201a87a24c3a2691c592143c6a333cdf652c357257981ea3450047647",
    "demos/08_option_vega/live_runtime.py": "706bd39717925f5e8376f958be6629187d770f5c02cf74bc0cb14a2c40772a92",
    "demos/08_option_vega/recovery.py": "c0708c564a6a3e181359be843341364789802024ced15fcb3fd7c41454dcbfc4",
    "demos/08_option_vega/run_live.py": "c66971265854b9d1b7ac80a63f59e2298d3ea1dc12e85f0221976d4a345b0ace",
    "demos/08_option_vega/strategy.py": "4961502b55a163d6e1cbe69d724771ddcc2beab6b5724efc7943251c411c25a6",
    "scripts/integration/options/__init__.py": "be65a34d8f3388d3252d04a55411d216618753783cbba6d4cf40f8aa24690b02",
    "scripts/integration/options/recording.py": "1d0ff5e5b14dd0223640bcfaacd94165ff3077d8e15d5594364ad8d711d39aeb",
    "scripts/integration/reference_data/calendar_export.py": "f6ac57245f7893d0cc7406173f573e466df143fb617de969c95f9c49c9e90555",
    "tests/run_option_vega_offline.py": "68202477ca13db8fe71094095f808a572150f190b24cd07d12273a6ad75e4495",
    "tests/test_calendars_compiled.py": "bb410784db982991974947cd52242e6800d46e0eecc55b8f57e1c3316abf3c54",
    "tests/test_index_query_feed.py": "d70678bf8e91c1f640f23e7f5952de4a8074e09313cbc56dfb8ee5609aea5f26",
    "tests/test_live_readiness.py": "68be26427fa88d3821f1aa520c84749756f1793a77c96aad1d0214f19f2a0540",
    "tests/test_live_recovery.py": "3d536ee48fbcc9fda1a91de95d8f65b2350e1fec00ed8eebe5295db8062101c6",
    "tests/test_option_recording_acceptance.py": "add809360c7b5c9f334e881f5df96d439dbbe5c1857ac2960ccdb3cbf484111d",
    "tests/test_option_vega_assembly.py": "0a36d914135988794c063c92d04aecdd91f048fe442cb90f024b294c938b3b23",
    "tests/test_option_vega_live.py": "1382d882350796814fe07d5fcfba5a21a11292abbfc1cae66f39ca615c992208",
    "tests/test_trading_calendar.py": "bde86802148881d8fda9aa0020ec2e05c35fbf134fa0bc0f7fa23b9da0991ff4",
    "bomber/framework/market/stream/health.py": "9db27606e1292e9e5a35c23ea2a32de466a0622effd960868e68d6a44353b8d1",
    "bomber/framework/market/stream/ctp/feed.py": "32fb53a788c4697e603c60f6184450815ac9e30d93d30dfe3e12a1a84320f9a1",
    "bomber/framework/dataprep/features/query.py": "3a94d8bc0735e64e60da485a1ea3683438c78946c8f6f2ffcffa4f9ef256523a",
    "tests/test_ctp_depth_health.py": "6ed1585e89550a26d05e79c3cf2438b8f2f497f5499ba1a0b1ad6ce83845bbc4",
    "demos/08_option_vega/replay_test.py": "725692fbf24f300a9c190459672b465924d3a8564a8af1646cf434a583e393fb",
    "tests/test_option_replay_test.py": "e40ee8fc098889e726e9b62c854194fa021bce36e5adbe5817b42221983937bd",
    "demos/08_option_vega/execution_audit.py": "87a3a2723b2342c1942b611eb02bee656c8585c19e5f8ce7d6578856d69818d2",
    "tests/test_option_execution_audit.py": "b1c2e4257a2fdafda2c5682c642fae057e8abbc17fe59131ae087ec603e61442",
}


def inspect_feature_deployment(root, expected, retired=()):
    """只读代码/配置字节；先识别混版，不导入原生组件或待测Demo。"""
    sources = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if (root / relative).is_file() else None for relative in expected}
    return {"sources": sources,
        "mismatches": [{"path": relative, "expected": digest, "actual": sources[relative]}
            for relative, digest in expected.items() if sources[relative] != digest],
        "retired_files": [relative for relative in retired if (root / relative).exists()]}


CASES = {
    "feature-datahub": ("tests.run_feature_datahub",),
    "feature-query-live": ("unittest", "tests.test_index_query_feed"),
    "option-vega": ("unittest", "tests.test_demo_option_vega", "tests.test_option_vega_live"),
    "option-vega-live": ("unittest", "tests.test_option_vega_assembly", "tests.test_option_replay_test", "tests.test_option_execution_audit"),
    "option-calendar": ("unittest", "tests.test_calendars_compiled", "tests.test_trading_calendar", "tests.test_option_recording_acceptance"),
    "execution-composition": ("tests.test_execution_composition",),
    "ema-recovery": ("tests.test_main_ema_recovery",),
    "continuous": ("tests.test_main_ema_continuous",),
    "live-common": ("unittest", "tests.test_live_common", "tests.test_live_channels", "tests.test_live_recovery", "tests.test_execution_composition", "tests.test_live_readiness", "tests.test_ctp_depth_health"),
    "cross-section": ("tests.test_cross_section_live",),
    "signal": ("tests.test_demo_cross_section_signal",),
    "entry": ("tests.run_main_ema_live",),
    "routing": ("tests.run_main_ema_simnow", "--stage", "routing"),
    "ctp": ("tests.run_main_ema_simnow", "--stage", "ctp"),
    "assembly": ("tests.test_strategy_assembly",),
    "dispatch": ("tests.run_execution_dispatch",),
    "attribution": ("tests.test_position_attribution",),
    "ctp-autosave": ("tests.run_p4_ctp_autosave",),
    "ctp-recovery": ("tests.run_p4_ctp_recovery",),
    "ctp-driver": ("tests.run_p4_ctp_driver",),
}
SOURCES = (
    "bomber/framework/datahub/features/schema.py",
    "bomber/framework/dataprep/features/raw.py",
    "bomber/framework/dataprep/features/definition.py",
    "bomber/framework/dataprep/features/query.py",
    "bomber/framework/dataprep/features/configs/features.json",
    "bomber/framework/dataprep/features/configs/features.example.json",
    "tests/test_feature_definitions.py",
    "bomber/framework/dataprep/features/dolphin_query.py",
    "bomber/framework/market/stream/feature_query.py",
    "bomber/framework/trader/runtime/live/channels/features.py",
    "tests/test_dolphindb_index_features.py",
    "tests/test_index_query_feed.py",
    "scripts/integration/features/index_readonly.py",
    "scripts/integration/features/read_only.py",
    "bomber/framework/datahub/features/contracts.py",
    "bomber/framework/datahub/features/store.py",
    "bomber/framework/datahub/features/alignment.py",
    "bomber/framework/datahub/features/frames.py",
    "bomber/framework/datahub/features/replay.py",
    "bomber/framework/datahub/features/__init__.py",
    "bomber/framework/dataprep/features/base.py",
    "bomber/framework/dataprep/features/codec.py",
    "bomber/framework/dataprep/features/file.py",
    "bomber/framework/dataprep/features/database.py",
    "bomber/framework/dataprep/features/factory.py",
    "bomber/framework/dataprep/features/service.py",
    "bomber/framework/dataprep/features/samples.py",
    "bomber/framework/dataprep/features/__init__.py",
    "bomber/framework/datahub/temporal.py",
    "bomber/framework/datahub/__init__.py",
    "bomber/framework/dataprep/sources/policy.py",
    "tests/run_feature_datahub.py",
    "tests/test_feature_datahub.py",
    "demos/08_option_vega/assembly.py",
    "demos/08_option_vega/strategy.py",
    "demos/08_option_vega/vega_signal.py",
    "demos/08_option_vega/run_backtest.py",
    "demos/08_option_vega/run_live.py",
    "demos/08_option_vega/live_runtime.py",
    "demos/08_option_vega/recovery.py",
    "demos/08_option_vega/checkpoint.py",
    "bomber/framework/market/frames.py",
    "bomber/framework/market/stream/aggregation.py",
    "bomber/framework/dataprep/live_options.py",
    "bomber/framework/dataprep/metadata.py",
    "bomber/framework/trader/runner.py",
    "bomber/framework/trader/execution/ctp/planner.py",
    "bomber/framework/trader/execution/ctp/limit_planner.py",
    "bomber/framework/trader/execution/builders.py",
    "bomber/framework/trader/runtime/live/channels/dolphin.py",
    "tests/test_option_vega_live.py",
    "tests/test_option_vega_assembly.py",
    "tests/run_option_vega_offline.py",
    "bomber/framework/trader/execution/policy.py",
    "bomber/framework/trader/execution/fanout.py",
    "bomber/framework/trader/execution/account.py",
    "bomber/framework/trader/execution/model.py",
    "bomber/framework/trader/execution/__init__.py",
    "bomber/framework/trader/runtime/live/local.py",
    "bomber/framework/trader/runtime/live/channels/ctp_market.py",
    "tests/test_execution_composition.py",
    "bomber/framework/trader/template.py",
    "bomber/framework/trader/portfolio.py",
    "bomber/framework/trader/execution/planner.py",
    "bomber/framework/trader/runtime/live/recovery.py",
    "bomber/framework/trader/runtime/live/channels/ctp_recovery.py",
    "demos/01_main_ema/recovery.py",
    "tests/test_live_recovery.py",
    "tests/test_main_ema_recovery.py",
    "scripts/integration/ctp_binance/run.py",
    "bomber/framework/trader/runtime/__init__.py",
    "bomber/framework/trader/runtime/direct.py",
    "bomber/framework/trader/runtime/live/__init__.py",
    "bomber/framework/trader/runtime/live/channels/__init__.py",
    "bomber/framework/trader/runtime/live/channels/binance.py",
    "bomber/framework/trader/runtime/live/channels/ctp.py",
    "bomber/framework/trader/runtime/live/contracts.py",
    "bomber/framework/trader/runtime/live/health.py",
    "bomber/framework/trader/runtime/live/profiles.py",
    "bomber/framework/trader/runtime/live/runtime.py",
    "tests/test_live_channels.py",

    "bomber/framework/trader/runtime/live_policy.py",
    "bomber/framework/trader/runtime/ctp_setup.py",
    "bomber/framework/trader/runtime/ctp_account.py",
    "bomber/framework/dataprep/live_cache.py",
    "bomber/framework/dataprep/live_cached_role.py",
    "bomber/framework/dataprep/live_portfolio.py",
    "bomber/framework/trader/live_roles.py",
    "bomber/framework/market/stream/aggregation.py",
    "tests/test_live_common.py",
    "demos/01_main_ema/continuous.py",
    "demos/01_main_ema/checkpoint.py",
    "demos/01_main_ema/shfe_cu_2026.json",
    "bomber/framework/trader/runtime/trading_sessions.py",
    "tests/test_main_ema_continuous.py",
    "bomber/framework/market/basic/base.py",

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
    "tests/run_p4_ctp_autosave.py", "tests/run_p4_ctp_recovery.py", "tests/run_p4_ctp_driver.py",
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
    report = {"python": sys.executable, "project": str(ROOT), "status": "running", "modules": [], "sources": {}}
    print(f"离线验收 Python={sys.executable} project={ROOT} report={path}", flush=True)
    try:
        if not args.only or FEATURE_DEPLOYMENT_GROUPS.intersection(args.only):
            from tests.run_feature_datahub import EXPECTED_SOURCE_HASHES, RETIRED_SOURCES, ACCEPTANCE_VERSION
            check = inspect_feature_deployment(ROOT, {**EXPECTED_SOURCE_HASHES, **FEATURE_RUNTIME_HASHES}, RETIRED_SOURCES)
            report["feature_deployment"] = {"version": ACCEPTANCE_VERSION, **check}
            report["sources"].update(check["sources"])
            for mismatch in check["mismatches"]:
                print(f"版本不一致: {mismatch['path']} expected={mismatch['expected']} actual={mismatch['actual']}", flush=True)
            if check["retired_files"]:
                print("须移除退役代码: " + ", ".join(check["retired_files"]), flush=True)
            if check["mismatches"] or check["retired_files"]:
                raise RuntimeError("特征/08交付版本不一致；先同步所列源码/配置及清理旧文件，再复测")
        for relative in SOURCES:
            if relative not in report["sources"]:
                source = ROOT / relative
                report["sources"][relative] = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
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
        recovery = import_module("demos.04_cross_section.recovery")
        common_recovery = import_module("bomber.framework.trader.runtime.live.channels.ctp_recovery")
        if not issubclass(recovery.ResumeLifecycle, common_recovery.CtpResumeLifecycle):
            raise RuntimeError("部署版本不一致：04/recovery.py仍为旧恢复流程；同步"
                "doc/evidence/checksums/PF2026101037_REV2_SHA256.txt所列文件并核对摘要后再测试")
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
