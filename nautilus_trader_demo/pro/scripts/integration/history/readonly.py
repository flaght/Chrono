"""固定真实合约历史分钟只读探针；不连接CTP、不预热EMA、不写交易状态。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from dotenv import load_dotenv

from bomber.framework.dataprep.history import (
    DolphinDbHistoryConfig, HistoryProviderFactory, HistoryRequest, HistoryError)
from bomber.framework.dataprep.sources import DolphinDbReferenceConfig, DataSourcePurpose


def probe(config, request, *, expected_bars=None, factory=HistoryProviderFactory):
    provider = factory.create_for(DataSourcePurpose.LIVE_RECOVERY, "dolphindb", config)
    try:
        provider.open()
        bars = provider.read(request)
    finally:
        provider.close()
    if not bars:
        raise HistoryError("查询未返回完整分钟；核对真实合约大小写、日期和源字段")
    if expected_bars is not None and len(bars) != expected_bars:
        raise HistoryError(f"分钟条数不符：预期{expected_bars}，实际{len(bars)}")
    if any(any(getattr(bar, field) is None for field in ("open", "high", "low", "close")) for bar in bars):
        raise HistoryError("原始行情只读核验要求完整OHLC")
    def sample(bar):
        return {"instrument_id": bar.instrument_id, "ts_event": bar.ts_event,
            "minute_start": datetime.fromtimestamp((bar.ts_event - 59_999_999_999) // 1000000000,
                timezone.utc).isoformat(),
            **{field: str(getattr(bar, field)) for field in ("open", "high", "low", "close")}}
    return {"status": "passed", "purpose": "fixed_contract_raw_history_readonly",
        "database": config.database, "table": config.table, "instrument": request.instrument_id,
        "start_ns": request.start_ns, "end_ns": request.end_ns, "bars": len(bars),
        "expected_bars": expected_bars, "first": sample(bars[0]), "last": sample(bars[-1]),
        "price_basis": "raw", "volume_policy": "not_mapped_cumulative_source",
        "ema_warmed": False, "orders_submitted": 0}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instrument", required=True, help="完整真实合约，例如IM2611.CFFEX")
    parser.add_argument("--start", required=True, help="含时区的分钟起点，包含")
    parser.add_argument("--end", required=True, help="含时区的分钟起点，不包含")
    parser.add_argument("--expected-bars", type=int)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not args.connect:
        parser.error("须显式--connect允许连接行情数据库")
    if not 1 <= args.timeout <= 120 or (args.expected_bars is not None and not 1 <= args.expected_bars <= 10000):
        parser.error("超时须为1至120秒，预期分钟条数须为1至10000")
    try:
        moments = [datetime.fromisoformat(value) for value in (args.start, args.end)]
        if any(moment.tzinfo is None or moment.second or moment.microsecond for moment in moments):
            raise ValueError()
        args.request = HistoryRequest(args.instrument,
            int(moments[0].timestamp()) * 1000000000, int(moments[1].timestamp()) * 1000000000)
        if moments[1] > datetime.now(timezone.utc):
            raise ValueError()
    except ValueError:
        parser.error("起止须为已结束、含时区的完整分钟边界且end晚于start")
    if args.report is not None and args.report.exists():
        parser.error("报告已存在，保留原证据并指定新路径")
    return args


def write_report(path, result):
    with Path(path).open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def main(argv=None):
    args = parse_args(argv)
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    data = json.loads(args.config.read_text())
    config = DolphinDbHistoryConfig(
        connection=DolphinDbReferenceConfig.from_env(read_timeout_seconds=args.timeout), **data)
    result = probe(config, args.request, expected_bars=args.expected_bars)
    if args.report:
        write_report(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
