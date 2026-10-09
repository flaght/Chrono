"""标准化静态合约资料，不创建原生交易合约。"""
from __future__ import annotations

from decimal import Decimal
import re

from bomber.framework.datahub.basic import positive_decimal
from bomber.framework.datahub.future_basic import FutureBasic
from bomber.framework.datahub.option_basic import OptionBasic

from .catalog import symbol
from .contracts import InstrumentSpec
from .session import fail, read_feather

VENUES = {"XSGE": "SHFE", "SHFE": "SHFE", "XDCE": "DCE", "DCE": "DCE",
          "XZCE": "CZCE", "CZCE": "CZCE", "XSIE": "INE", "INE": "INE",
          "CCFX": "CFFEX", "XCFX": "CFFEX", "CFFEX": "CFFEX",
          "XGFE": "GFEX", "GFEX": "GFEX"}
REQUIRED_BASIC = frozenset({"symbol", "code", "exchangeCD", "contMultNum",
                           "minChgPriceNum", "listDate", "lastTradeDate"})


def positive(value, name):
    return positive_decimal(value, name)


def contract_date(value, code, field):
    import pandas as pd
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("Empty date")
        return stamp.date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{code}: {field}日期无效") from exc


def validate_basic(frame, required=REQUIRED_BASIC):
    if not frame.columns.is_unique or set(required) - set(frame.columns):
        fail("MISSING_FIELD", f"Basic data missing fields or duplicate columns: {sorted(set(required) - set(frame.columns))}")
    return frame


def venue_of(row):
    venue = VENUES.get(str(row["exchangeCD"]).strip().upper())
    if venue is None:
        fail("INVALID_METADATA", f"Unsupported exchange {row['exchangeCD']}")
    return venue


def symbol_of(row):
    return str(row["symbol"]).strip().split(".")[0]


def venue(basic, product):
    validate_basic(basic)
    rows = basic.loc[basic.code.astype(str).str.strip().str.upper().eq(product.upper())]
    if rows.empty:
        fail("INVALID_METADATA", f"fut_basic has no product {product}")
    candidates = {venue_of(row) for _, row in rows.iterrows()}
    if len(candidates) != 1:
        fail("CONFLICTING_METADATA", f"{product} maps to several venues: {candidates}")
    return candidates.pop()


def selected_products(basic, products):
    return {product: venue(basic, product) for product in products}


def contract_rows(basic, keys):
    validate_basic(basic)
    result = {}
    for key in sorted(keys):
        code, separator, requested = key.partition(".")
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", code)
        if not match or separator and not requested:
            fail("INVALID_METADATA", f"Target must be a real futures contract: {key}")
        rows = basic.loc[basic.symbol.map(symbol).eq(symbol(code))]
        if len(rows) != 1:
            fail("CONFLICTING_METADATA", f"Need one fut_basic row for {code}, found {len(rows)}")
        row = rows.iloc[0]
        if separator and requested.upper() != venue_of(row):
            fail("IDENTITY_MISMATCH", f"{key}: exchange differs from fut_basic")
        if str(row["code"]).strip().upper() != match[1].upper():
            fail("IDENTITY_MISMATCH", f"{key}: product differs from fut_basic")
        result[key] = row
    return result


def future_spec(row, *, require_execution=True):
    _require_static_publication(row)
    return future_spec_from_basic(parse_future_terms(row, require_execution=require_execution))


def parse_future_terms(row, *, require_execution=True):
    """仅解析条款值，供静态入口及版本化Provider共用。

    available_ns/source_version不是条款值，不在这里判断可见性；在线调用方
    必须用原始记录的发布时间门控快照，不能仅凭该模型决定可交易。
    """
    code = symbol(row["symbol"])
    listed = contract_date(row["listDate"], code, "listDate")
    last = contract_date(row["lastTradeDate"], code, "lastTradeDate")
    if listed > last:
        fail("INVALID_METADATA", f"{code}: listing after last trading day")
    return FutureBasic(
        symbol=code, product=str(row["code"]).strip().upper(),
        exchange=venue_of(row), currency=str(row.get("currencyCD", "CNY")).strip().upper(),
        list_date=listed, last_trade_date=last,
        price_increment=positive(row["minChgPriceNum"], code + "/tick") if require_execution else None,
        multiplier=positive(row["contMultNum"], code + "/multiplier") if require_execution else None,
    )


def future_spec_from_basic(basic):
    """将已校验的条款值转换为执行规格；时间门控由资料Provider承担。"""
    if not isinstance(basic, FutureBasic):
        raise TypeError("期货规格转换需要 FutureBasic")
    return InstrumentSpec("future", basic.symbol, basic.product, basic.exchange,
        basic.currency, basic.list_date, basic.last_trade_date,
        tick=basic.price_increment, multiplier=basic.multiplier)


def _require_static_publication(row):
    """旧示例使用静态模型，因此拒绝带发布版本的资料。"""
    import pandas as pd
    if any(field in row and pd.notna(row[field]) and str(row[field]).strip()
           for field in ("available_ns", "source_version")):
        fail("UNSUPPORTED_CAPABILITY", "Versioned contract facts need an as-of metadata provider")


def load_futures_basic(path, products=None, *, require_execution=True):
    frame = read_feather(path)
    required = REQUIRED_BASIC if require_execution else {"symbol", "code", "exchangeCD", "listDate", "lastTradeDate"}
    validate_basic(frame, required)
    product_set = None if products is None else {p.upper() for p in products}
    result = {}
    for row in frame.to_dict("records"):
        if product_set is not None and str(row["code"]).strip().upper() not in product_set:
            continue
        info = future_spec(row, require_execution=require_execution)
        if info.symbol in result:
            fail("CONFLICTING_METADATA", f"Duplicate futures contract {info.symbol}", source=path)
        result[info.symbol] = info
    return result


def load_cffex_futures(path, product, *, require_execution=False):
    frame = read_feather(path)
    validate_basic(frame, REQUIRED_BASIC if require_execution else
        {"symbol", "code", "exchangeCD", "listDate", "lastTradeDate"})
    rows = {}
    for raw in frame.to_dict("records"):
        if str(raw["code"]).strip().upper() != product.upper():
            continue
        code = symbol(raw["symbol"])
        if not re.fullmatch(re.escape(product.upper()) + r"\d{4}", code):
            continue
        info = future_spec(raw, require_execution=require_execution)
        if info.venue != "CFFEX":
            fail("IDENTITY_MISMATCH", f"{code}: expected CFFEX", source=path)
        if code in rows:
            fail("CONFLICTING_METADATA", f"Duplicate contract {code}", source=path)
        row = dict(raw)
        row["listDate"], row["lastTradeDate"] = info.listed, info.last_trade
        if require_execution:
            row["minChgPriceNum"], row["contMultNum"] = info.tick, info.multiplier
        rows[code] = row
    return rows


def load_options_basic(path, product, index_code, tick_override=None, *, kinds=("C", "P"),
                       require_month_dates=True, require_currency=False):
    import pandas as pd
    frame = read_feather(path)
    if not frame.columns.is_unique:
        fail("MISSING_FIELD", "opt_basic重复列", source=path)
    if "symbol" not in frame and "Code" in frame:
        frame = frame.rename(columns={"Code": "symbol"})
    if {"symbol", "Code"} <= set(frame) and not frame.Code.map(symbol).eq(frame.symbol.map(symbol)).all():
        fail("CONFLICTING_METADATA", "opt_basic的Code与symbol不一致", source=path)
    required = {"symbol", "contractType", "strikePrice", "contMultNum", "varTicker",
                "exchangeCD", "listDate", "lastTradeDate", "expDate"}
    validate_basic(frame, required)
    result, expiries = {}, {}
    for row in frame.to_dict("records"):
        code = symbol(row["symbol"])
        match = re.fullmatch(re.escape(product.upper()) + r"(\d{4})-([CP])-(\d+)", code)
        if not match or match[2] not in kinds:
            continue
        _require_static_publication(row)
        kind = match[2]
        allowed = ("C", "CO", "CALL", "认购") if kind == "C" else ("P", "PO", "PUT", "认沽")
        if str(row["contractType"]).strip().upper() not in allowed:
            fail("CONFLICTING_METADATA", f"{code}: option kind conflict", source=path)
        if venue_of(row) != "CFFEX" or str(row["varTicker"]).strip().zfill(6) != index_code:
            fail("IDENTITY_MISMATCH", f"{code}: exchange or underlying mismatch", source=path)
        currency = str(row.get("currencyCD", "CNY")).strip().upper()
        if require_currency and currency != "CNY":
            fail("INVALID_METADATA", f"{code}: expected CNY", source=path)
        # 新条款优先使用 tickNum，旧字段仅在前一字段缺失或为空时兼容。
        tick = None
        for field in ("tickNum", "minChgPriceNum", "price_increment"):
            value = row.get(field)
            if value is not None and not pd.isna(value):
                tick = value
                break
        if tick is None:
            tick = tick_override
        if tick is None:
            raise ValueError(f"{code}: opt_basic缺少最小变动价位，请提供 tickNum（兼容 minChgPriceNum、price_increment）")
        strike = positive(row["strikePrice"], code + "/strike")
        if strike != Decimal(match[3]):
            fail("CONFLICTING_METADATA", f"{code}: strike differs from symbol", source=path)
        listed = contract_date(row["listDate"], code, "listDate")
        last = contract_date(row["lastTradeDate"], code, "lastTradeDate")
        expiry = contract_date(row["expDate"], code, "expDate")
        month = "20" + match[1]
        if not listed <= last <= expiry:
            fail("INVALID_METADATA", f"{code}: invalid lifecycle", source=path)
        if require_month_dates and (last.strftime("%Y%m") != month or expiry.strftime("%Y%m") != month):
            fail("CONFLICTING_METADATA", f"{code}: 合约月份与到期日期不一致", source=path)
        if require_month_dates and month in expiries and expiries[month] != last:
            fail("CONFLICTING_METADATA", f"{code}: 同月合约最后交易日期不一致", source=path)
        expiries[month] = last
        basic = OptionBasic(
            symbol=code, exchange="CFFEX", currency=currency, underlying=index_code,
            option_kind=kind, strike=strike,
            multiplier=positive(row["contMultNum"], code + "/multiplier"),
            list_date=listed, last_trade_date=last, expiry_date=expiry,
            price_increment=positive(tick, code + "/tick"),
        )
        info = InstrumentSpec("option", basic.symbol, product.upper(), basic.exchange,
            basic.currency, basic.list_date, basic.last_trade_date,
            basic.price_increment, basic.multiplier, basic.expiry_date, kind,
            basic.strike, basic.underlying, month)
        if code in result and result[code] != info:
            fail("CONFLICTING_METADATA", f"{code}: conflicting static versions", source=path)
        result[code] = info
    if not result:
        raise ValueError(f"没有{product}期权资料")
    return result
