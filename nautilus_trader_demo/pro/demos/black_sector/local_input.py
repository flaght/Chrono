"""第五类本地样本装配：Feather只在此层出现，策略/DataHub接口不依赖它。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from datahub.role_prices import ObservedClose, RoleAssignment, RolePriceStore
from datahub.sector_roles import SectorRoleAssignment, SectorRoleStore


_FILE = re.compile(r"^([A-Za-z]+\d+)_(\d{8})\.feather$")
_ROLE_COLUMNS = {"trade_date", "code", "main", "second", "recent", "far"}


@dataclass(frozen=True)
class LoadedSectorResearch:
    store: SectorRoleStore
    day_end_ns: tuple[tuple[date, int], ...]
    bars_dir: Path


def _day(value: object) -> date:
    import pandas as pd

    return pd.Timestamp(value).date()


def _ns(value: object, timezone: str) -> int:
    import pandas as pd

    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone)
    return int(stamp.tz_convert("UTC").value)


def load_sector_research(
    *, bars_dir: str | Path, contract_struct_path: str | Path,
    timezone: str = "Asia/Shanghai",
    signal_products: tuple[str, ...],
    signal_role: str,
    execution_product: str,
    execution_role: str,
    end_day: date | None = None,
) -> LoadedSectorResearch:
    """读取真实收盘价，独立计算三个品种次主力连续因子。

    每个交易日使用之前最近的角色表；换约因子只由旧、新合约在来源日
    的同日收盘价构成。缺少价格时显式失败，不能默认因子为1。
    """
    import pandas as pd
    import pyarrow as pa

    if not signal_products or len(set(signal_products)) != len(signal_products):
        raise ValueError("信号品种不能为空或重复")
    if execution_product not in signal_products or not signal_role or not execution_role:
        raise ValueError("执行品种必须在信号品种中，角色名不能为空")
    root = Path(bars_dir)
    files: list[tuple[Path, str, date]] = []
    for path in root.rglob("*.feather"):
        match = _FILE.match(path.name)
        if match and match.group(1).rstrip("0123456789").upper() in signal_products:
            label = match.group(2)
            trading_day = date.fromisoformat(f"{label[:4]}-{label[4:6]}-{label[6:]}")
            if end_day is None or trading_day <= end_day:
                files.append((path, match.group(1).lower(), trading_day))
    if not files:
        raise FileNotFoundError(f"没有{signal_products}真实合约Bar: {root}")
    roles = pd.read_feather(contract_struct_path)
    if _ROLE_COLUMNS - set(roles.columns):
        raise ValueError(f"角色表缺少列: {sorted(_ROLE_COLUMNS - set(roles.columns))}")
    roles["code"] = roles["code"].astype(str).str.strip().str.upper()
    roles = roles.loc[roles["code"].isin(signal_products)].copy()
    roles["source_day"] = roles["trade_date"].map(_day)
    roles = roles.sort_values("source_day").drop_duplicates(["source_day", "code"], keep="last")

    closes: dict[str, list[ObservedClose]] = {key: [] for key in signal_products}
    first_by_day: dict[date, int] = {}
    last_by_day: dict[date, int] = {}
    for path, symbol, trading_day in sorted(files):
        product = symbol.rstrip("0123456789").upper()
        try:
            frame = pd.read_feather(path, columns=["datetime", "close"])
            time_key = "datetime"
        except (KeyError, ValueError, pa.ArrowInvalid):
            try:
                frame = pd.read_feather(path, columns=["timestamp", "close"])
                time_key = "timestamp"
            except (KeyError, ValueError, pa.ArrowInvalid) as exc:
                raise ValueError(f"{path}缺少datetime/timestamp或close") from exc
        if frame.empty:
            continue
        stamps = [_ns(item, timezone) for item in frame[time_key]]
        first_by_day[trading_day] = min(first_by_day.get(trading_day, stamps[0]), min(stamps))
        last_by_day[trading_day] = max(last_by_day.get(trading_day, stamps[-1]), max(stamps))
        latest = max(range(len(stamps)), key=stamps.__getitem__)
        closes[product].append(ObservedClose(
            symbol, trading_day, stamps[latest], Decimal(str(frame["close"].iloc[latest])),
        ))
    shared_days = tuple(sorted(first_by_day))
    product_stores: dict[str, RolePriceStore] = {}
    for product in signal_products:
        source = roles.loc[roles["code"] == product]
        assignments: list[RoleAssignment] = []
        for day in shared_days:
            prior = source.loc[source["source_day"] < day]
            if prior.empty:
                continue
            row = prior.iloc[-1]
            assignments.append(RoleAssignment(
                day, row["source_day"], first_by_day[day], first_by_day[day],
                {"main": str(row["main"]).lower(), "secondary": str(row["second"]).lower(),
                 "near": str(row["recent"]).lower(), "far": str(row["far"]).lower()},
            ))
        if not assignments:
            raise ValueError(f"{product}没有可用的上一交易日角色记录")
        product_stores[product] = RolePriceStore(
            tuple(assignments), tuple(closes[product]),
            missing_roll_policy="previous_common", max_anchor_lookback_trading_days=1,
        )
    records: list[SectorRoleAssignment] = []
    for day in shared_days:
        effective = first_by_day[day]
        try:
            selected = {key: product_stores[key].assignment_at(effective)
                        for key in signal_products}
            factors = {key: product_stores[key].factor_at(effective, signal_role)[1]
                       for key in signal_products}
        except LookupError as exc:
            # 日期不完整不可静默沿用昨日角色。正式回测必须修复样本缺口。
            raise ValueError(f"{day}品种角色/信号复权不可用: {exc}") from exc
        if any(row.trading_day != day for row in selected.values()):
            raise ValueError(f"{day}缺少当天生效的完整角色")
        source_days = {row.source_day for row in selected.values()}
        if len(source_days) != 1:
            raise ValueError(f"{day}三个品种角色表来源日不一致: {sorted(source_days)}")
        if execution_role not in selected[execution_product].contracts:
            raise ValueError(f"{day}/{execution_product}缺少执行角色{execution_role}")
        records.append(SectorRoleAssignment(
            day, source_days.pop(), effective, effective,
            {key: selected[key].contracts for key in signal_products},
            {key: {signal_role: factors[key]} for key in signal_products},
        ))
    return LoadedSectorResearch(
        SectorRoleStore(tuple(records)), tuple((day, last_by_day[day]) for day in shared_days), root,
    )
