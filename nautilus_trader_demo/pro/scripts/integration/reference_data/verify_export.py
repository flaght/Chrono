"""纯本地校验参考导出与可见快照；不配置或连接数据库／柜台。"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from bomber.framework.dataprep.live_references import LiveFuturesReferences
from bomber.framework.dataprep.reference_freshness import ReferenceFreshnessPolicy
from bomber.framework.dataprep.sources.artifacts import ExportedReferenceSource


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trading-day", required=True)
    parser.add_argument("--expected-source-day", required=True)
    parser.add_argument("--products", nargs="+", required=True)
    parser.add_argument("--as-of-ns", type=int, help="默认采用包的稳定观测时刻，不能回填此前可见性")
    parser.add_argument("--reference-max-source-age-days", type=int, default=14)
    args = parser.parse_args(argv)
    try:
        day, source_day = (datetime.strptime(value, "%Y%m%d").date()
            for value in (args.trading_day, args.expected_source_day))
        if day.strftime("%Y%m%d") != args.trading_day or source_day.strftime("%Y%m%d") != args.expected_source_day:
            raise ValueError("日期须为YYYYMMDD")
        freshness = ReferenceFreshnessPolicy(source_day, args.reference_max_source_age_days)
    except ValueError as error:
        parser.error(str(error))
    with ExportedReferenceSource(args.bundle, as_of_ns=0) as source:
        as_of = args.as_of_ns if args.as_of_ns is not None else source.manifest["observed_ns"]
        source.as_of_ns = as_of
        refs = LiveFuturesReferences(source, products=tuple(args.products), trading_day=day,
            started_ns=as_of, factor_date_basis="source", factor_availability="explicit",
            clock_ns=lambda: as_of, freshness=freshness)
        snapshot = refs.snapshot(as_of)
        result = {"status": "passed", "mode": "offline_reference", "as_of_ns": as_of,
            "trading_day": str(day), "source_day": str(snapshot.source_day),
            "contracts": {product: dict(values) for product, values in snapshot.contracts.items()},
            "cumulative_factors": {product: {role: str(value) for role, value in values.items()}
                for product, values in snapshot.cumulative_factors.items()},
            "export_manifest": source.manifest, "reference_manifest": refs.manifest}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
