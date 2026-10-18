"""按数据集映射字段；Code 的含义不能跨表通用推断。"""
from datetime import date
import re

from .base import ReferenceBatch, ReferenceDataset


FIELD_ALIASES = {
    ReferenceDataset.FUTURES_BASIC: {"Code": "symbol", "contractObject": "code", "date": "listDate"},
    ReferenceDataset.OPTIONS_BASIC: {"Code": "symbol", "contractObject": "code", "date": "listDate"},
    ReferenceDataset.ADJUSTMENT_FACTORS: {"Code": "symbol", "contractObject": "code", "date": "trade_date"},
    ReferenceDataset.CONTRACT_STRUCTURE: {"Code": "code", "trade_date": "date"},
}
DATE_FIELDS = {
    ReferenceDataset.FUTURES_BASIC: "listDate",
    ReferenceDataset.OPTIONS_BASIC: "listDate",
    ReferenceDataset.ADJUSTMENT_FACTORS: "trade_date",
    ReferenceDataset.CONTRACT_STRUCTURE: "date",
}
REQUIRED = {
    ReferenceDataset.FUTURES_BASIC: {"symbol", "code", "listDate", "exchangeCD", "lastTradeDate", "contMultNum", "minChgPriceNum"},
    ReferenceDataset.OPTIONS_BASIC: {"symbol", "listDate", "exchangeCD", "lastTradeDate", "contractType", "strikePrice", "contMultNum", "varTicker", "expDate"},
    ReferenceDataset.ADJUSTMENT_FACTORS: {"symbol", "code", "trade_date", "pcr_cumfactor"},
    ReferenceDataset.CONTRACT_STRUCTURE: {"code", "date"},
}


def normalize_frame(frame, dataset, query, *, source):
    import pandas as pd
    dataset = ReferenceDataset(dataset)
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("数据源须返回列名唯一的 DataFrame")
    frame = frame.copy()
    for old, new in FIELD_ALIASES[dataset].items():
        if old not in frame:
            continue
        if new in frame:
            if new in {"listDate", "trade_date", "date"}:
                same = pd.to_datetime(frame[old], errors="raise").eq(pd.to_datetime(frame[new], errors="raise"))
            else:
                same = frame[old].astype(str).str.strip().str.upper().eq(frame[new].astype(str).str.strip().str.upper())
            if not same.all():
                raise ValueError(f"来源字段 {old} 与标准字段 {new} 冲突")
            frame = frame.drop(columns=old)
        else:
            frame = frame.rename(columns={old: new})
    required = REQUIRED[dataset]
    if dataset is ReferenceDataset.ADJUSTMENT_FACTORS and "cumulative_factor" in frame:
        required = required - {"pcr_cumfactor"} | {"cumulative_factor", "role"}
    if required - set(frame):
        raise ValueError(f"{dataset.value} 缺少字段: {sorted(required - set(frame))}")
    for column in ("code", "symbol"):
        if column in frame:
            values = frame[column]
            if values.isna().any() or values.astype(str).str.strip().eq("").any():
                raise ValueError(f"{column} 不能为空")
            frame[column] = values.astype(str).str.strip().str.upper()
    # 期权表不制造 code：只在显式品种筛选时按真实合约前缀选择。
    if query.products:
        if "code" in frame:
            products = frame.code
        elif dataset is ReferenceDataset.OPTIONS_BASIC:
            products = frame.symbol.str.extract(r"^([A-Z]+)\d", expand=False)
            if products.isna().any():
                raise ValueError("期权代码无法按品种前缀筛选，请按明确 symbols 查询")
        else:
            raise ValueError("数据集缺少品种身份")
        frame = frame.loc[products.isin(query.products)].copy()
    if query.symbols:
        if "symbol" not in frame:
            raise ValueError("期限结构应按 products 查询")
        frame = frame.loc[frame.symbol.isin(query.symbols)].copy()
    day_field = DATE_FIELDS[dataset]
    days = pd.to_datetime(frame[day_field], errors="raise")
    if days.isna().any():
        raise ValueError(f"{day_field} 不能为空")
    frame[day_field] = days.dt.date
    if query.start_date:
        frame = frame.loc[frame[day_field] >= query.start_date]
    if query.end_date:
        frame = frame.loc[frame[day_field] <= query.end_date]
    if query.active_on:
        if dataset not in {ReferenceDataset.FUTURES_BASIC, ReferenceDataset.OPTIONS_BASIC}:
            raise ValueError("active_on 仅用于合约基础资料")
        last = pd.to_datetime(frame.lastTradeDate, errors="raise")
        if last.isna().any():
            raise ValueError("lastTradeDate 不能为空")
        frame = frame.loc[(frame[day_field] <= query.active_on) & (last.dt.date >= query.active_on)]
    # 保留可选发布时间和版本字段；普通数据库日期并不是发布时间。
    rows = tuple({key: None if pd.isna(value) else value for key, value in row.items()}
                 for row in frame.to_dict("records"))
    return ReferenceBatch(dataset, rows, tuple(frame.columns), source)
