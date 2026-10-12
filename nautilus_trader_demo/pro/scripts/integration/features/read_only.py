"""按JSON数据集配置进行只读探针、标准化读取和文件往返验收。"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("须为ISO时间") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("时间须含显式时区，例如+08:00")
    return int(parsed.timestamp())*1_000_000_000 + parsed.microsecond*1000


def main(argv=None):
    from bomber.framework.dataprep.features import DEFAULT_FEATURE_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--schema-version", type=int, default=1)
    parser.add_argument("--key", required=True)
    parser.add_argument("--database")
    parser.add_argument("--table")
    parser.add_argument("--start", type=timestamp)
    parser.add_argument("--end", type=timestamp)
    parser.add_argument("--as-of", type=timestamp)
    parser.add_argument("--availability", choices=("observed", "source-time", "update-time"))
    parser.add_argument("--timestamp-label", choices=("start", "end", "inclusive-end"))
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parents[3]/"tests/results")
    args = parser.parse_args(argv)
    if not args.connect:
        parser.error("须显式--connect")
    if not args.inspect and (args.start is None or args.end is None):
        parser.error("读取须明确--start及--end；仅Schema用--inspect")
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3]/".env")
    from bomber.framework.datahub.features import FeatureHub, FeatureQuery, FeatureRequirement
    from bomber.framework.dataprep.features import (bundle_from_definition, FeatureProviderFactory,
        DolphinDbRawFeatureProvider, FeatureRequest, FileFeatureProvider, export_features)
    from bomber.framework.dataprep.sources.dolphindb import DolphinDbReferenceConfig
    report_dir = args.report_dir/f"feature-readonly-{time.time_ns()}"
    report_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "dataset_id": args.dataset, "schema_version": args.schema_version,
        "key": args.key, "config": str(args.config.resolve())}
    provider, code = None, 0
    try:
        content = args.config.read_bytes()
        definition = json.loads(content)
        backends = {d["source"]["backend"] for d in definition["datasets"]}
        connections = {}
        if backends.intersection({"dolphindb", "dolphindb-raw"}):
            connections["dolphindb"] = DolphinDbReferenceConfig.from_env(read_timeout_seconds=args.timeout)
        if "mongodb" in backends:
            connections["mongodb"] = {"uri": os.getenv("MONGODB_URI"), "timeout_seconds": args.timeout}
        bundle = bundle_from_definition(definition, connections, base_dir=args.config.resolve().parent)
        binding = bundle.binding(args.dataset, args.schema_version)
        registry, config = bundle.registry, binding.source_config
        overrides = {name: getattr(args, name) for name in ("database", "table", "availability")
            if getattr(args, name) is not None}
        if overrides.get("availability") == "update-time":
            overrides["availability"] = "source-time"
        if binding.backend == "dolphindb-raw":
            if args.timestamp_label is not None:
                overrides["event_time"] = replace(config.event_time, label=args.timestamp_label)
            config = replace(config, **overrides)
        elif overrides or args.timestamp_label is not None:
            raise ValueError("CLI原表覆盖只适用于dolphindb-raw；其他来源请编辑配置")
        if getattr(config, "availability", None) == "observed" and args.as_of is not None:
            raise ValueError("observed无法查询过去可用性；历史as-of须有源发布时间证据")
        provider = FeatureProviderFactory.create_for(
            "offline_feature" if binding.backend == "file" else "live_feature", binding.backend,
            config, registry, **binding.provider_options)
        report.update(backend=binding.backend, config_sha256=hashlib.sha256(content).hexdigest(),
            schema_fingerprint=binding.spec.fingerprint, availability=getattr(config, "availability", "standard-envelope"))
        if args.inspect:
            if not isinstance(provider, DolphinDbRawFeatureProvider):
                raise ValueError("原表Schema探针仅用于dolphindb-raw；其他来源使用区间读取")
            provider.transport.open()
            report["schema"] = provider.inspect(sample_rows=0)
            report["schema"]["sample"] = provider.inspect(sample_key=args.key)["sample"]
        else:
            provider.open()
            cutoff = args.as_of if args.as_of is not None else time.time_ns()
            request = FeatureRequest(args.dataset, args.schema_version, (args.key,), args.start, args.end,
                cutoff, max_rows=args.max_rows)
            batch = provider.read(request)
            if not batch.records:
                raise ValueError("指定区间没有可见数据；不能按接入通过处理")
            hub = FeatureHub(registry)
            hub.store.publish(batch.records)
            latest_event = max(r.ts_event for r in batch.records)
            completed = time.time_ns()
            visible_at = args.as_of if args.as_of is not None else completed
            requirement = FeatureRequirement("value", args.dataset, args.schema_version, args.key)
            snapshot = hub.snapshot((requirement,), FeatureQuery(latest_event, visible_at))
            report.update(records=len(batch.records), query_completed_ns=completed,
                first_event_ns=min(r.ts_event for r in batch.records), last_event_ns=latest_event,
                latest_record=asdict(snapshot.latest("value")))
            if args.export:
                report["export_manifest"] = export_features(args.export, registry, batch.records)
                with FileFeatureProvider(args.export, registry) as file:
                    replay = file.read(replace(request, as_of_ns=visible_at))
                if replay.records != batch.records:
                    raise ValueError("数据库导出与文件读取记录不一致")
                report.update(export=str(args.export.resolve()), file_roundtrip="passed")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        code = 1
    finally:
        if provider is not None:
            try:
                provider.close()
            except Exception as error:
                report.update(status="failed", close_error=str(error))
                code = 1
        path = report_dir/"summary.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
        print(f"详细报告: {path}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
