# -*- coding: utf-8 -*-
"""CTP 五类监控数据的标准表结构和字段转换。"""

import datetime


CHINA_TZ = datetime.timezone(datetime.timedelta(hours=8))

DIRECTION_MAP = {"0": "买", "1": "卖", "2": "多", "3": "空"}
OFFSET_MAP = {
    "0": "开仓", "1": "平仓", "2": "强平", "3": "平今",
    "4": "平昨", "5": "强减", "6": "本地强平",
}
ORDER_STATUS_MAP = {
    "0": "全部成交", "1": "部分成交还在队列", "2": "部分成交不在队列",
    "3": "未成交还在队列", "4": "未成交不在队列", "5": "已撤单",
    "a": "未知", "b": "尚未触发", "c": "已触发",
}

FUNDS_COLUMNS = [
    "UpdateTime", "TradingDay", "ProductID", "AccountID", "BrokerID",
    "GatewayName", "Balance", "DynamicEquity", "Margin", "Available",
    "CloseProfit", "PositionProfit", "Commission", "FrozenMargin",
    "Deposit", "Withdraw",
]

POSITIONS_COLUMNS = [
    "UpdateTime", "TradingDay", "ProductID", "AccountID", "BrokerID",
    "GatewayName", "ExchangeID", "InstrumentID", "PosiDirection",
    "DirectionName", "PositionDate", "HedgeFlag", "Position", "YdPosition",
    "TodayPosition", "LongFrozen", "ShortFrozen", "OpenVolume", "CloseVolume",
    "OpenAmount", "CloseAmount", "PositionCost", "UseMargin", "Commission",
    "CloseProfit", "PositionProfit", "PreSettlementPrice", "SettlementPrice",
]

ORDERS_COLUMNS = [
    "UpdateTime", "TradingDay", "ProductID", "AccountID", "BrokerID",
    "GatewayName", "ExchangeID", "InstrumentID", "OrderSysID", "OrderLocalID",
    "FrontID", "SessionID", "OrderRef", "Direction", "DirectionName",
    "CombOffsetFlag", "OffsetName", "CombHedgeFlag", "OrderPriceType",
    "LimitPrice", "VolumeTotalOriginal", "VolumeTraded", "VolumeTotal",
    "OrderStatus", "OrderStatusName", "OrderSubmitStatus", "StatusMsg",
    "InsertDate", "InsertTime", "ActiveTime", "CancelTime", "SequenceNo",
    "BrokerOrderSeq", "RequestID",
]

TRADES_COLUMNS = [
    "UpdateTime", "TradingDay", "ProductID", "AccountID", "BrokerID",
    "GatewayName", "ExchangeID", "InstrumentID", "TradeID", "OrderSysID",
    "OrderLocalID", "Direction", "DirectionName", "OffsetFlag", "OffsetName",
    "HedgeFlag", "Price", "Volume", "TradeDate", "TradeTime", "TradeType",
    "PriceSource", "SequenceNo", "BrokerOrderSeq",
]

POSITION_DETAILS_COLUMNS = [
    "UpdateTime", "TradingDay", "ProductID", "AccountID", "BrokerID",
    "GatewayName", "ExchangeID", "InstrumentID", "TradeID", "Direction",
    "DirectionName", "HedgeFlag", "OpenDate", "Volume", "OpenPrice",
    "SettlementPrice", "LastSettlementPrice", "Margin", "MarginRateByMoney",
    "MarginRateByVolume", "CloseVolume", "CloseAmount", "TimeFirstVolume",
    "CombInstrumentID",
]

TABLE_COLUMNS = {
    "funds": FUNDS_COLUMNS,
    "positions": POSITIONS_COLUMNS,
    "orders": ORDERS_COLUMNS,
    "trades": TRADES_COLUMNS,
    "position_details": POSITION_DETAILS_COLUMNS,
}


def china_now():
    return datetime.datetime.now(CHINA_TZ).replace(tzinfo=None)


def _text(value):
    return "" if value is None else str(value).strip()


def _number(raw, name, default=0.0):
    value = raw.get(name, default)
    return default if value is None else value


def _base(raw, gateway_name, config):
    return {
        "UpdateTime": china_now(),
        "TradingDay": _text(raw.get("TradingDay")),
        "ProductID": _text(config.get("product_id")),
        "AccountID": _text(
            raw.get("AccountID") or raw.get("InvestorID") or config.get("account_id")
        ),
        "BrokerID": _text(raw.get("BrokerID") or config.get("broker_id")),
        "GatewayName": gateway_name,
    }


def _fill(row, raw, columns):
    for column in columns:
        if column not in row:
            row[column] = raw.get(column, "")
    return row


def normalize_fund(raw, gateway_name, config):
    row = _base(raw, gateway_name, config)
    balance = float(_number(raw, "Balance"))
    position_profit = float(_number(raw, "PositionProfit"))
    row.update({
        "Balance": balance,
        "DynamicEquity": balance,
        "Margin": _number(raw, "CurrMargin"),
        "Available": _number(raw, "Available"),
        "CloseProfit": _number(raw, "CloseProfit"),
        "PositionProfit": position_profit,
        "Commission": _number(raw, "Commission"),
        "FrozenMargin": _number(raw, "FrozenMargin"),
        "Deposit": _number(raw, "Deposit"),
        "Withdraw": _number(raw, "Withdraw"),
    })
    return _fill(row, raw, FUNDS_COLUMNS)


def normalize_position(raw, gateway_name, config):
    row = _base(raw, gateway_name, config)
    direction = _text(raw.get("PosiDirection"))
    row.update({
        "PosiDirection": direction,
        "DirectionName": DIRECTION_MAP.get(direction, direction),
    })
    return _fill(row, raw, POSITIONS_COLUMNS)


def normalize_order(raw, gateway_name, config):
    row = _base(raw, gateway_name, config)
    direction = _text(raw.get("Direction"))
    offset_flag = _text(raw.get("CombOffsetFlag"))
    offset = offset_flag[:1]
    status = _text(raw.get("OrderStatus"))
    row.update({
        "Direction": direction,
        "DirectionName": DIRECTION_MAP.get(direction, direction),
        "CombOffsetFlag": offset_flag,
        "OffsetName": OFFSET_MAP.get(offset, offset),
        "OrderStatus": status,
        "OrderStatusName": ORDER_STATUS_MAP.get(status, status),
    })
    return _fill(row, raw, ORDERS_COLUMNS)


def normalize_trade(raw, gateway_name, config):
    row = _base(raw, gateway_name, config)
    direction = _text(raw.get("Direction"))
    offset = _text(raw.get("OffsetFlag"))[:1]
    row.update({
        "Direction": direction,
        "DirectionName": DIRECTION_MAP.get(direction, direction),
        "OffsetFlag": offset,
        "OffsetName": OFFSET_MAP.get(offset, offset),
    })
    return _fill(row, raw, TRADES_COLUMNS)


def normalize_position_detail(raw, gateway_name, config):
    row = _base(raw, gateway_name, config)
    direction = _text(raw.get("Direction"))
    row.update({
        "Direction": direction,
        "DirectionName": DIRECTION_MAP.get(direction, direction),
    })
    return _fill(row, raw, POSITION_DETAILS_COLUMNS)


NORMALIZERS = {
    "funds": normalize_fund,
    "positions": normalize_position,
    "orders": normalize_order,
    "trades": normalize_trade,
    "position_details": normalize_position_detail,
}


def row_key(table_name, row):
    gateway = _text(row.get("GatewayName"))
    if table_name == "funds":
        return "|".join([gateway, _text(row.get("AccountID"))])
    if table_name == "positions":
        return "|".join([
            gateway, _text(row.get("ExchangeID")), _text(row.get("InstrumentID")),
            _text(row.get("DirectionName") or row.get("PosiDirection")),
        ])
    if table_name == "orders":
        system_id = _text(row.get("OrderSysID"))
        if not system_id:
            system_id = "|".join([
                _text(row.get("FrontID")), _text(row.get("SessionID")),
                _text(row.get("OrderRef")),
            ])
        return "|".join([gateway, _text(row.get("ExchangeID")), system_id])
    if table_name == "trades":
        return "|".join([
            gateway, _text(row.get("ExchangeID")), _text(row.get("TradeID")),
        ])
    return "|".join([
        gateway, _text(row.get("ExchangeID")), _text(row.get("TradeID")),
        _text(row.get("OpenDate")), _text(row.get("Direction")),
    ])

