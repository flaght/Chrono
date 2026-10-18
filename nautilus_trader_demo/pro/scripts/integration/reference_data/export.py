"""DolphinDB参考资料稳定导出；不连接行情或交易柜台。"""
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from bomber.framework.dataprep.sources import (
    DataSourcePurpose, DolphinDbReferenceConfig, ReferenceDataset, ReferenceQuery, ReferenceSourceFactory)
from bomber.framework.dataprep.sources.artifacts import export_reference_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--products", nargs="+", required=True)
    parser.add_argument("--start-day", required=True)
    parser.add_argument("--end-day", required=True)
    parser.add_argument("--database")
    parser.add_argument("--datasets", nargs="+", choices=tuple(value.value for value in ReferenceDataset),
        default=["futures_basic", "adjustment_factors", "contract_structure"])
    parser.add_argument("--visibility", choices=("explicit", "observed"), default="explicit",
        help="explicit须有真实available_ns；observed仅用于观测后回放，不能回测更早日期")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args(argv)
    if not args.connect:
        parser.error("须显式--connect允许导出连接数据库")
    if args.destination.exists():
        parser.error("导出文件已存在，不覆盖旧版本")
    try:
        days = [datetime.strptime(value, "%Y%m%d").date() for value in (args.start_day, args.end_day)]
        if any(day.strftime("%Y%m%d") != raw for day, raw in zip(days, (args.start_day, args.end_day))):
            raise ValueError("日期须为YYYYMMDD")
        if days[0] > days[1]:
            raise ValueError("起始日晚于结束日")
    except ValueError as error:
        parser.error(str(error))
    queries = {}
    for name in args.datasets:
        dataset = ReferenceDataset(name)
        # 基础条款的日期是上市日，不按导出区间过滤上市日。
        queries[dataset] = ReferenceQuery(products=tuple(args.products),
            **({"active_on": days[1]} if dataset in {
                ReferenceDataset.FUTURES_BASIC, ReferenceDataset.OPTIONS_BASIC} else
                {"start_date": days[0], "end_date": days[1]}))
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    config = DolphinDbReferenceConfig.from_env(database=args.database, read_timeout_seconds=args.timeout)
    with ReferenceSourceFactory.create_for(DataSourcePurpose.EXPORT, "dolphindb", config) as source:
        result = export_reference_bundle(source, queries, args.destination, visibility=args.visibility)
    print(f"导出完成: {args.destination.resolve()} SHA256={result['sha256']} visibility={args.visibility}")


if __name__ == "__main__":
    main()
