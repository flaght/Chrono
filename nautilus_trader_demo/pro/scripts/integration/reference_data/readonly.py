"""读取DolphinDB参考表并核验在线期货快照；不连接CTP、不报单。"""
from datetime import datetime
import argparse
import json
from pathlib import Path
import time

from dotenv import load_dotenv

from bomber.framework.datahub.option_basic import OptionBasic
from bomber.framework.dataprep.live_references import LiveFuturesReferences, live_factor_policy
from bomber.framework.dataprep.sources import (
    DolphinDbReferenceConfig, ReferenceQuery, ReferenceSourceFactory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--trading-day", required=True, help="已核对的YYYYMMDD交易日")
    parser.add_argument("--products", nargs="+", required=True)
    parser.add_argument("--database", help="默认DDB_REFERENCE_DATABASE或dfs://bomber_daily")
    parser.add_argument("--factor-date-basis", choices=("source", "trading"), default="source",
                        help="原始日表默认source；已对齐到TD适用日的数据显式用trading")
    parser.add_argument("--factor-availability", choices=("source-day-end", "aligned", "explicit", "observed-on-read"))
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--option-product", help="可选核验opt_basic，例如MO")
    parser.add_argument("--report", type=Path, help="可选保存脱敏的资料核验结果")
    args = parser.parse_args(argv)
    if not args.connect:
        parser.error("须显式 --connect 允许连接参考数据库")
    try:
        day = datetime.strptime(args.trading_day, "%Y%m%d").date()
        args.factor_availability = live_factor_policy(args.factor_date_basis, args.factor_availability)
    except ValueError:
        parser.error("交易日须为YYYYMMDD，source用source-day-end/explicit/observed-on-read，trading用aligned/explicit")
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    config = DolphinDbReferenceConfig.from_env(database=args.database, read_timeout_seconds=args.timeout)
    with ReferenceSourceFactory.create("dolphindb", config) as source:
        references = LiveFuturesReferences(source, products=tuple(args.products),
            trading_day=day, started_ns=time.time_ns(), factor_date_basis=args.factor_date_basis,
            factor_availability=args.factor_availability)
        snapshot = references.snapshot(time.time_ns())
        result = {"manifest": references.manifest, "source_day": str(snapshot.source_day),
            "contracts": {p: dict(values) for p, values in snapshot.contracts.items()},
            "cumulative_factors": {p: dict(values) for p, values in snapshot.cumulative_factors.items()},
            "instruments": {p: {role: {"symbol": spec.symbol, "venue": spec.venue,
                "listed": str(spec.listed), "last_trade": str(spec.last_trade),
                "tick": str(spec.tick), "multiplier": str(spec.multiplier)}
                for role, spec in values.items()} for p, values in references.instrument_specs.items()}}
        if args.option_product:
            batch = source.options_basic(ReferenceQuery(products=(args.option_product,), active_on=day))
            if not batch.rows:
                raise ValueError("opt_basic没有请求品种的有效期权")
            # 校验全部请求条款，而非只检查展示用的首行。
            info = [OptionBasic.from_mapping(row) for row in batch.rows]
            result["options"] = {"rows": len(info), "sha256": batch.fingerprint,
                                  "first_symbol": info[0].symbol, "source": batch.source}
        output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(output + "\n", encoding="utf-8")
        print(output, flush=True)
        print("参考数据库只读核验通过；未连接交易柜台、未报单", flush=True)


if __name__ == "__main__":
    main()
