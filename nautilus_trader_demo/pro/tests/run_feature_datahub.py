"""独立GAP08验收：源码加载路径/摘要、无网络契约组、历史DataHub兼容组。"""
import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_VERSION = "PF2026101208_REV2"
RETIRED_SOURCES = ("bomber/framework/datahub/features/index.py", "bomber/framework/dataprep/features/index.py",
    "bomber/framework/market/stream/dolphin/index.py", "bomber/framework/trader/runtime/live/channels/index.py")
# 与本次被测代码一起交付；远程代码部署不依赖doc/目录。
# 入口自身SHA只记录审计，避免内置摘要的自引用。
EXPECTED_SOURCE_HASHES = {
    "bomber/framework/datahub/__init__.py": "35c231a4208a091de6e95bb0397c337888165485da041e900e0dbb922292150d",
    "bomber/framework/datahub/features/__init__.py": "dc6c00b797e2b71025dcf4cc7b0a995f631dbcf6bff27d5e969a033d187b095c",
    "bomber/framework/datahub/features/alignment.py": "18274df424302b4253fdbc250906d6c86de482fb55c4aa9021b923ad8610b099",
    "bomber/framework/datahub/features/contracts.py": "dd2155227778d6b19135a47daf445ce4621fe8749a3c237b3dbadcc3b74d1409",
    "bomber/framework/datahub/features/frames.py": "ad31385053862a057144e96e079eebb2f04db78cbd4cc9549f66743c3f04cdcc",
    "bomber/framework/datahub/features/replay.py": "0988404db8e9b2da2e6ee3936e4a0076e7e63262b8b50cbd945e61d7696a13b4",
    "bomber/framework/datahub/features/schema.py": "fa1ff693062957fcf7a1cfa5172710229bc185466f154661c993c8af059627df",
    "bomber/framework/datahub/features/store.py": "4df163553bd64490172e8c6071707183c5d5a81911b6e8a97832625da66345d4",
    "bomber/framework/datahub/temporal.py": "62463da79376ef793031597d2a3f62fcc1c0d469cb6a7f6dbd533c6d89519968",
    "bomber/framework/dataprep/features/__init__.py": "23fc8b3ea3719e09ff1316eb129b03e4212f33a119979efbc89d3e69a5573dee",
    "bomber/framework/dataprep/features/base.py": "f739063883e78f835996ad0058642cf10b71d8c3c8c8d77b0a1937056f683855",
    "bomber/framework/dataprep/features/codec.py": "af47f15be55f68a04aebc92d611637f754dd20ea859457e6689be99fda74902b",
    "bomber/framework/dataprep/features/configs/features.example.json": "3103ca44d0e2f194d46f691f4a76c206203e7062ed8a9c63abc092bbaeda317b",
    "bomber/framework/dataprep/features/configs/features.json": "eba898958f39a4235c4f62751d3e510ac0682fa1eac570a4bca6f43a25967470",
    "bomber/framework/dataprep/features/database.py": "127e6da8bcf137e2042085327baef34e6dfd2eb0223e766a8233d48546b9a9c3",
    "bomber/framework/dataprep/features/definition.py": "fe71ab0ffe97cc92831ee74cad13c8013ec587427df882b6d207fa414cd1a720",
    "bomber/framework/dataprep/features/dolphin_query.py": "89d4be92072aec75b9eecd03f9c4b0c7879b87ee9cd82b1eb413eee7e9da1615",
    "bomber/framework/dataprep/features/factory.py": "32c383f4b3221116f11635c8734c8f17ffc2bc880ee9dd21ecd8772a8e1f8df6",
    "bomber/framework/dataprep/features/file.py": "188daa6879a6e3fce86aa930ac63376f975c96c15f151ee7201f4b2c6afaed9d",
    "bomber/framework/dataprep/features/query.py": "3a94d8bc0735e64e60da485a1ea3683438c78946c8f6f2ffcffa4f9ef256523a",
    "bomber/framework/dataprep/features/raw.py": "64669b104d16dae8b4e4f595f90d195e5f8cce3decaa61bed9e740283cbee1b7",
    "bomber/framework/dataprep/features/samples.py": "3d82f143b6034d23f70b2dccf6a2ccbc8cdf9658a2a500173c1c082171efae5d",
    "bomber/framework/dataprep/features/service.py": "c70092bcce0ee5189853075e207bc090fb240e31b31bf9dec1e760121b6a44a1",
    "bomber/framework/dataprep/sources/dolphindb.py": "7eafbf43f08ac0184e0fd9c31c805e7c9b114ffa0e4f4e6c7ad99b5467b14e8a",
    "bomber/framework/dataprep/sources/policy.py": "33697be3613a893a2efda11ca83ed6e808ba73599024cb347c355cfee4c68016",
    "bomber/framework/market/frames.py": "984d7a11ec0352b12013513962542601881b3a0ec9a51053b1d0366f52e14d75",
    "scripts/integration/features/__init__.py": "47b4fef8a3a5722516e725699b06a7cc07810f24ab32936a42ac9b1258f3338b",
    "scripts/integration/features/index_readonly.py": "2d34503c97f333929ba41a72faf6a4bc06cf8963712581427cc8bf247d7c94e2",
    "scripts/integration/features/read_only.py": "f4f591965459151a318e13cd6f2d942584f7911588003895b467464be864349b",
    "tests/test_dolphindb_index_features.py": "72f8107575170d4151131364051c26685a3ea82408cddd0cb3ba53b25216cb62",
    "tests/test_feature_acceptance_entry.py": "45706cff7e93a3a3fa2c74ee54d9f56f390e3c7bc2913ce736229872c95d30b5",
    "tests/run_cross_section_acceptance.py": "809f7f77badc7457c19ddb55e62b3d934aceb612e39df051c1deff59f4b79169",
    "tests/test_feature_datahub.py": "06ea20c4810696711aa16560dbf0fcea7102a23c729d1aea391f60d7f6f1b63a",
    "tests/test_feature_definitions.py": "66336d4061d813c3b55b8f2d84c258f28a0356c51a08089865ee9d2ed41745d2",
}
SOURCES = tuple(sorted(
    [str(path.relative_to(ROOT)) for folder in (
        ROOT / "bomber/framework/datahub/features", ROOT / "bomber/framework/dataprep/features")
     for path in folder.glob("*.py")]
    + ["bomber/framework/dataprep/features/configs/features.json",
       "bomber/framework/dataprep/features/configs/features.example.json"]
    + ["bomber/framework/datahub/temporal.py", "bomber/framework/datahub/__init__.py",
       "bomber/framework/dataprep/sources/policy.py", "bomber/framework/market/frames.py",
       "tests/test_feature_datahub.py", "tests/run_feature_datahub.py",
       "tests/test_feature_acceptance_entry.py", "tests/test_dolphindb_index_features.py",
       "bomber/framework/dataprep/sources/dolphindb.py",
       "tests/test_feature_definitions.py", "scripts/integration/features/read_only.py",
       "tests/run_cross_section_acceptance.py",
       "scripts/integration/features/index_readonly.py", "scripts/integration/features/__init__.py"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="GAP08无网络验收，不连接数据库或柜台")
    parser.add_argument("--only", choices=("features", "definitions", "index", "legacy", "entry"), action="append")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "tests/results")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--manifest", type=Path,
        help="可选外部源码摘要；默认使用Python入口内置摘要，无需部署文档")
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("timeout须为正整数")
    report_path = args.report_dir / f"feature-datahub-{time.time_ns()}"
    report_path.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "python": sys.executable, "project": str(ROOT),
        "modules": [], "sources": {}, "acceptance_version": ACCEPTANCE_VERSION}
    def save():
        (report_path / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"GAP08离线验收 Python={sys.executable} project={ROOT} report={report_path}", flush=True)
    try:
        retired = [path for path in RETIRED_SOURCES if (ROOT / path).exists()]
        report["retired_source_files"] = retired
        if retired:
            print("须移除退役代码: " + ", ".join(retired), flush=True)
            raise RuntimeError("指数专用模块已退役；清理上列旧文件后再测")
        if args.manifest is None:
            expected = dict(EXPECTED_SOURCE_HASHES)
            report["manifest"] = f"embedded:{ACCEPTANCE_VERSION}"
        else:
            expected = {}
            for line in args.manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                digest, relative = line.split(maxsplit=1)
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest) or relative in expected:
                    raise RuntimeError("部署摘要格式无效或路径重复")
                expected[relative] = digest
            report["manifest"] = str(args.manifest.resolve())
        entry_source = ROOT / "tests/run_feature_datahub.py"
        report["entry_sha256"] = hashlib.sha256(entry_source.read_bytes()).hexdigest() if entry_source.is_file() else None
        checked = sorted((set(SOURCES) - {"tests/run_feature_datahub.py"}) | {
            path for path in expected if path.endswith(".py")
            and path.startswith(("bomber/framework/datahub/features/", "bomber/framework/dataprep/features/"))})
        for path in checked:
            source = ROOT / path
            report["sources"][path] = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        mismatches = [path for path in checked if report["sources"][path] != expected.get(path)]
        report["deployment_mismatches"] = mismatches
        if mismatches:
            for path in mismatches:
                print(f"版本不一致: {path} expected={expected.get(path)} actual={report['sources'][path]}", flush=True)
            raise RuntimeError("运行机文件与交付不一致；先同步所列代码文件，再复测")
        report["loaded_sources"] = {}
        for name in ("bomber.framework.datahub.features", "bomber.framework.dataprep.features",
                     "bomber.framework.market.frames", "tests.test_feature_datahub"):
            module = import_module(name)
            actual = Path(module.__file__).resolve()
            if not actual.is_relative_to(ROOT):
                raise RuntimeError(f"加载了其他项目源码: {name}")
            digest = hashlib.sha256(actual.read_bytes()).hexdigest()
            report["loaded_sources"][name] = {"path": str(actual), "sha256": digest}
            if digest != expected.get(str(actual.relative_to(ROOT))):
                raise RuntimeError(f"已加载模块摘要不匹配: {name}")
            print(f"加载路径 {name}={actual} SHA256={digest}", flush=True)
    except Exception as error:
        report.update(status="failed", deployment_error=str(error), deployment_error_type=type(error).__name__)
        print(f"部署/加载预检失败 [{type(error).__name__}]: {error}", flush=True)
        print(f"详细报告: {report_path / 'summary.json'}", flush=True)
        save()
        return 2
    groups = {"features": ("unittest", "tests.test_feature_datahub"),
        "definitions": ("unittest", "tests.test_feature_definitions"),
        "index": ("unittest", "tests.test_dolphindb_index_features"),
        "legacy": ("unittest", "tests.test_datahub_option_basic", "tests.test_basic_alignment"),
        "entry": ("unittest", "tests.test_feature_acceptance_entry")}
    for name, modules in groups.items():
        if args.only and name not in args.only:
            continue
        started = time.monotonic()
        try:
            process = subprocess.run([sys.executable, "-u", "-m", *modules], cwd=ROOT,
                timeout=args.timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            code, output = process.returncode, process.stdout
        except subprocess.TimeoutExpired as error:
            raw = error.stdout or ""
            code, output = 124, (raw.decode(errors="replace") if isinstance(raw, bytes) else raw) + "\n超时\n"
        (report_path/f"{name}.log").write_text(output, encoding="utf-8")
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        report["modules"].append({"name": name, "exit_code": code, "seconds": round(time.monotonic()-started, 3)})
        report["status"] = "failed" if code else "running"
        save()
        if code:
            return 1
    # run_datahub使用顶层函数；unittest不收集它，额外显式运行原入口。
    if not args.only or "legacy" in args.only:
        try:
            process = subprocess.run([sys.executable, "-u", "-m", "tests.run_datahub"], cwd=ROOT,
                timeout=args.timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except subprocess.TimeoutExpired:
            report.update(status="failed", legacy_d0="timeout")
            save()
            return 1
        (report_path/"legacy-d0.log").write_text(process.stdout, encoding="utf-8")
        print(process.stdout, end="", flush=True)
        report["modules"].append({"name": "legacy-d0", "exit_code": process.returncode})
        if process.returncode:
            report["status"] = "failed"
            save()
            return 1
    report["status"] = "passed"
    save()
    print("选定GAP08离线组通过；真实数据库与策略消费场景须分别验收。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
