"""第四类示例的本地数据准备，仅服务当前用户提供的Feather测试文件。

不属于DataHub核心或其正式Provider协议；未来文件、数据库及在线来源
可在装配层转换为标准记录，无需继承这里的文件读取代码。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from datahub.role_prices import (
    ObservedClose, ResearchDataUnavailable, RoleAssignment, RolePriceStore,
)


_RB_FILE = re.compile(r"^(rb\d+)_(\d{8})\.feather$", re.IGNORECASE)


@dataclass(frozen=True)
class PcrAudit:
    trading_day: date
    source_day: date
    main_instrument: str
    calculated_single: Decimal | None
    source_day_symbol: str | None
    effective_day_symbol: str | None
    source_day_single: Decimal | None
    effective_day_single: Decimal | None
    source_day_cumulative: Decimal | None
    effective_day_cumulative: Decimal | None


@dataclass(frozen=True)
class LoadedRoleResearch:
    store: RolePriceStore
    audit: tuple[PcrAudit, ...]
    bar_count: int
    contract_count: int
    day_end_ns: tuple[tuple[date, int], ...]


def _timestamp_ns(value: Any, timezone: str) -> int:
    import pandas as pd

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("行情时间不能为空")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    return int(timestamp.tz_convert("UTC").value)


def _day(value: Any) -> date:
    import pandas as pd

    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("交易日不能为空")
    return result.date()


def _file_trading_day(filename: str) -> date:
    match = _RB_FILE.match(filename)
    if match is None:
        raise ValueError(f"RB Bar文件名不符合合约_交易日格式: {filename}")
    label = match.group(2)
    return date.fromisoformat(f"{label[:4]}-{label[4:6]}-{label[6:]}")


def load_rb_role_research(
    *,
    bars_dir: str | Path,
    contract_struct_path: str | Path,
    factors_path: str | Path,
    fut_basic_path: str | Path,
    timezone: str = "Asia/Shanghai",
) -> LoadedRoleResearch:
    """按每个实际交易日取上一交易日的角色记录，不用自然日减一天。

    pcr只做对账，不直接套给near/secondary/far；这三条序列按各自换约
    和真实合约的前日收盘价生成。对账的数值方向需由因子生产方确认。
    """

    import pandas as pd

    bars_root = Path(bars_dir)
    paths = sorted(path for path in bars_root.rglob("*.feather") if _RB_FILE.match(path.name))
    if not paths:
        raise FileNotFoundError(f"未找到RB真实合约Feather Bar: {bars_root}")

    roles = pd.read_feather(contract_struct_path)
    required_roles = {"trade_date", "code", "recent", "main", "second", "far"}
    if required_roles - set(roles.columns):
        raise ValueError(f"合约结构表缺少列: {sorted(required_roles - set(roles.columns))}")
    roles = roles.loc[roles["code"].astype(str).str.upper() == "RB"].copy()
    if roles.empty:
        raise ValueError("合约结构表没有RB记录")
    roles["source_day"] = roles["trade_date"].map(_day)
    roles = roles.sort_values("source_day").drop_duplicates("source_day", keep="last")

    factors = pd.read_feather(factors_path)
    factor_columns = {"trade_date", "code", "symbol", "pcr_factor", "pcr_cumfactor"}
    if factor_columns - set(factors.columns):
        raise ValueError(f"复权因子表缺少列: {sorted(factor_columns - set(factors.columns))}")
    factors = factors.loc[factors["code"].astype(str).str.upper() == "RB"].copy()
    factors["source_day"] = factors["trade_date"].map(_day)
    factors = factors.sort_values("source_day").drop_duplicates("source_day", keep="last")
    factor_by_day = {row.source_day: row for row in factors.itertuples(index=False)}

    basic = pd.read_feather(fut_basic_path)
    # 新版fut_basic只提供合约身份和乘数，不再包含最小变动价位。
    # 研究价构建不需要tick size；记录型Bar探针由装配参数显式提供。
    required_basic = {"code", "symbol", "exchangeCD", "contMultNum"}
    if required_basic - set(basic.columns):
        raise ValueError(f"期货合约基础表缺少列: {sorted(required_basic - set(basic.columns))}")
    rb_basic = basic.loc[basic["code"].astype(str).str.upper() == "RB"]
    known_contracts = set(rb_basic["symbol"].astype(str).str.lower())

    closes: list[ObservedClose] = []
    first_ns_by_day: dict[date, int] = {}
    last_ns_by_day: dict[date, int] = {}
    actual_contracts: set[str] = set()
    for path in paths:
        match = _RB_FILE.match(path.name)
        assert match is not None
        instrument = match.group(1).lower()
        # 文件名中的YYYYMMDD是交易日；Bar的date可能是前一自然日夜盘。
        file_trading_day = _file_trading_day(path.name)
        actual_contracts.add(instrument)
        # Bar目录可能包含本次四角色之外的新挂牌合约；研究价只需要其收盘
        # 数据，不应因fut_basic尚未收录这个无关合约而中断整批读取。
        # 真正进入角色映射的合约仍在下方严格校验，模拟Instrument另行核验。
        frame = pd.read_feather(path)
        time_column = "datetime" if "datetime" in frame else "timestamp"
        if time_column not in frame or "close" not in frame:
            raise ValueError(f"{path}缺少datetime/timestamp或close列")
        for row in frame.to_dict("records"):
            if "trading_day" in row and _day(row["trading_day"]) != file_trading_day:
                raise ValueError(f"{path}的trading_day列与文件交易日不一致")
            trading_day = file_trading_day
            ns = _timestamp_ns(row[time_column], timezone)
            close = ObservedClose(instrument, trading_day, ns, Decimal(str(row["close"])))
            closes.append(close)
            first_ns_by_day[trading_day] = min(first_ns_by_day.get(trading_day, ns), ns)
            last_ns_by_day[trading_day] = max(last_ns_by_day.get(trading_day, ns), ns)

    assignments: list[RoleAssignment] = []
    audit_rows: list[PcrAudit] = []
    for trading_day in sorted(first_ns_by_day):
        historical = roles.loc[roles["source_day"] < trading_day]
        if historical.empty:
            continue  # 第一天只用来提供前日价格，不能读取当天日终角色。
        row = historical.iloc[-1]
        source_day = row["source_day"]
        contracts = {
            "main": str(row["main"]).strip().lower(),
            "secondary": str(row["second"]).strip().lower(),
            "near": str(row["recent"]).strip().lower(),
            "far": str(row["far"]).strip().lower(),
        }
        if any(item not in known_contracts for item in contracts.values()):
            raise ValueError(f"{trading_day}角色表引用fut_basic未知合约: {contracts}")
        ns = first_ns_by_day[trading_day]
        # 表的来源日早于当前交易日，但实际发布时间文件里未提供。这里明确
        # 以首根Bar时刻作为最早可用时刻，不提前宣称夜盘开盘前即可查询。
        assignments.append(RoleAssignment(trading_day, source_day, ns, ns, contracts))
    if not assignments:
        raise ValueError("行情范围内没有可用的上一交易日RB角色记录")

    # 旧合约已到期而来源日缺价时，最多回看前一个有行情的交易日；只有
    # 旧、新合约在同一天都有真实收盘价才能锚定。仍缺价则记录缺口并
    # 跳过受影响的策略决策，绝不假设因子为1。
    store = RolePriceStore(
        tuple(assignments), tuple(closes),
        missing_roll_policy="previous_common",
        max_anchor_lookback_trading_days=1,
    )
    for assignment in assignments:
        source_item = factor_by_day.get(assignment.source_day)
        effective_item = factor_by_day.get(assignment.trading_day)
        try:
            single, _ = store.factor_at(assignment.effective_ns, "main")
        except ResearchDataUnavailable:
            single = None
        audit_rows.append(PcrAudit(
            assignment.trading_day,
            assignment.source_day,
            assignment.contracts["main"],
            single,
            None if source_item is None else str(source_item.symbol).lower(),
            None if effective_item is None else str(effective_item.symbol).lower(),
            None if source_item is None else Decimal(str(source_item.pcr_factor)),
            None if effective_item is None else Decimal(str(effective_item.pcr_factor)),
            None if source_item is None else Decimal(str(source_item.pcr_cumfactor)),
            None if effective_item is None else Decimal(str(effective_item.pcr_cumfactor)),
        ))
    eligible_days = {assignment.trading_day for assignment in assignments}
    day_ends = tuple((day, last_ns_by_day[day]) for day in sorted(eligible_days))
    return LoadedRoleResearch(store, tuple(audit_rows), len(closes), len(actual_contracts), day_ends)


# 兼容当前M4b装配脚本的导入名；此本地加载器仍只处理RB样本。
load_role_research = load_rb_role_research
