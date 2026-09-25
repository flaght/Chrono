"""按品种加载主力、次主力、远期真实合约 Bar 和上一交易日角色表。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from datahub.role_prices import ObservedClose, RoleAssignment, RolePriceStore

_FILE = re.compile(r"^([A-Za-z]+\d+)_(\d{8})\.feather$")
_ROLE_COLUMNS = {"trade_date", "code", "main", "second", "far"}
SIGNAL_ROLES = ("main", "secondary", "far")


@dataclass(frozen=True)
class LoadedRoleResearch:
    store: RolePriceStore
    day_end_ns: tuple[tuple[date, int], ...]
    bar_count: int
    file_count: int


def _day(value: object) -> date:
    import pandas as pd
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("角色表交易日不能为空")
    return stamp.date()


def _ns(value: object, timezone: str) -> int:
    import pandas as pd
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Bar 时间不能为空")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone)
    return int(stamp.tz_convert("UTC").value)


def load_role_research(*, bars_dir: Path, contract_struct_path: Path,
                       product: str, end_day: date,
                       timezone: str = "Asia/Shanghai") -> LoadedRoleResearch:
    """构建因果角色/复权价；外部累计因子不参与信号计算。"""
    import pandas as pd
    import pyarrow as pa

    product = product.strip().upper()
    if not product.isalpha():
        raise ValueError("品种代码必须是字母")
    roles = pd.read_feather(contract_struct_path)
    if _ROLE_COLUMNS - set(roles.columns):
        raise ValueError(f"角色表缺少列: {sorted(_ROLE_COLUMNS - set(roles.columns))}")
    roles = roles.loc[roles["code"].astype(str).str.strip().str.upper() == product].copy()
    if roles.empty:
        raise ValueError(f"角色表没有品种 {product}")
    roles["source_day"] = roles["trade_date"].map(_day)
    roles = roles.sort_values("source_day").drop_duplicates("source_day", keep="last")
    selected_symbols = {
        str(symbol).strip().lower()
        for symbol in roles.loc[roles["source_day"] < end_day, ["main", "second", "far"]].to_numpy().flat
        if pd.notna(symbol)
    }

    files: dict[tuple[date, str], Path] = {}
    for path in bars_dir.rglob("*.feather"):
        match = _FILE.fullmatch(path.name)
        if match is None:
            continue
        symbol = match.group(1).lower()
        if symbol.rstrip("0123456789").upper() != product:
            continue
        if symbol not in selected_symbols:
            continue
        label = match.group(2)
        day = date.fromisoformat(f"{label[:4]}-{label[4:6]}-{label[6:]}")
        if day > end_day:
            continue
        key = (day, symbol)
        if key in files:
            raise ValueError(f"重复真实合约 Bar 文件: {files[key]}, {path}")
        files[key] = path
    if not files:
        raise FileNotFoundError(f"{bars_dir} 中没有截至 {end_day} 的 {product} Bar")

    closes: list[ObservedClose] = []
    first_ns: dict[date, int] = {}
    last_ns: dict[date, int] = {}
    for (day, symbol), path in sorted(files.items()):
        try:
            frame = pd.read_feather(path, columns=["datetime", "close"])
            time_column = "datetime"
        except (KeyError, ValueError, pa.ArrowInvalid):
            try:
                frame = pd.read_feather(path, columns=["timestamp", "close"])
                time_column = "timestamp"
            except (KeyError, ValueError, pa.ArrowInvalid) as exc:
                raise ValueError(f"{path} 缺少 datetime/timestamp 或 close 列") from exc
        for timestamp, raw_close in zip(frame[time_column], frame["close"]):
            ns = _ns(timestamp, timezone)
            closes.append(ObservedClose(symbol, day, ns, Decimal(str(raw_close))))
            first_ns[day] = min(first_ns.get(day, ns), ns)
            last_ns[day] = max(last_ns.get(day, ns), ns)
    assignments = []
    for day in sorted(first_ns):
        prior = roles.loc[roles["source_day"] < day]
        if prior.empty:
            continue
        row = prior.iloc[-1]
        contracts = {"main": str(row["main"]).strip().lower(),
                     "secondary": str(row["second"]).strip().lower(),
                     "far": str(row["far"]).strip().lower()}
        if any(not symbol or symbol == "nan" for symbol in contracts.values()):
            raise ValueError(f"{day}/{product} 主力、次主力或远期真实合约不完整")
        assignments.append(RoleAssignment(day, row["source_day"],
                                          first_ns[day], first_ns[day], contracts,
                                          roles=SIGNAL_ROLES))
    if not assignments:
        raise ValueError(f"{product} 没有可用的上一交易日角色记录")
    store = RolePriceStore(
        tuple(assignments), tuple(closes),
        missing_roll_policy="previous_common", max_anchor_lookback_trading_days=1,
    )
    return LoadedRoleResearch(
        store, tuple((row.trading_day, last_ns[row.trading_day]) for row in assignments),
        len(closes), len(files),
    )
