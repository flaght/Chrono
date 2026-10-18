"""将角色文件转换为现有 DataHub 存储，不生成可交易价格。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from .bars import read_bar_frame
from .catalog import scan_bar_files
from .contracts import BarReadSpec
from .session import fail, read_feather

ROLE_COLUMNS = {"main": "main", "secondary": "second", "near": "recent", "far": "far"}


@dataclass(frozen=True)
class LoadedSectorResearch:
    store: object
    day_end_ns: tuple[tuple[date, int], ...]
    bars_dir: Path


@dataclass(frozen=True)
class LoadedRoleResearch:
    store: object
    day_end_ns: tuple[tuple[date, int], ...]
    bar_count: int
    file_count: int


def load_role_assignments(path, products, required_roles):
    import pandas as pd
    frame = read_feather(path)
    roles_by_product = (required_roles if isinstance(required_roles, dict)
                        else {product: tuple(required_roles) for product in products})
    needed_roles = {role for roles in roles_by_product.values() for role in roles}
    if not needed_roles <= ROLE_COLUMNS.keys():
        fail("INVALID_ROLE", f"Unknown roles: {needed_roles - ROLE_COLUMNS.keys()}", source=path)
    required = {"trade_date", "code"} | {ROLE_COLUMNS[role] for role in needed_roles}
    if not frame.columns.is_unique or required - set(frame):
        fail("MISSING_FIELD", f"Role table missing {sorted(required - set(frame))}", source=path)
    frame["code"] = frame.code.astype(str).str.strip().str.upper()
    frame = frame.loc[frame.code.isin(products)].copy()
    stamps = pd.to_datetime(frame.trade_date, errors="raise")
    if stamps.isna().any():
        fail("INVALID_TIMESTAMP", "Role source day is empty", source=path)
    frame["source_day"] = stamps.dt.date
    for product in products:
        subset = frame.loc[frame.code.eq(product)]
        if subset.empty:
            fail("REFERENCE_NOT_AVAILABLE", f"Role table has no {product}", source=path)
        columns = [ROLE_COLUMNS[role] for role in roles_by_product[product]]
        if "available_ns" in subset:
            columns.append("available_ns")
        for source_day, rows in subset.groupby("source_day"):
            if len(rows[columns].astype(str).drop_duplicates()) != 1:
                fail("CONFLICTING_METADATA", f"Conflicting role records {product}/{source_day}", source=path)
    return frame.sort_values(["source_day", "code"], kind="stable").drop_duplicates(["source_day", "code"], keep="last")


def _collect_closes(index, *, minute, time_spec):
    from bomber.framework.datahub.role_prices import ObservedClose
    closes, first, last = [], {}, {}
    for key, path in sorted(index.items()):
        result = read_bar_frame(path, time_spec, key)
        frame = result.frame
        # ObservedClose 没有可用时间字段，因此拒绝延迟发布的锚点，
        # 避免预计算因子泄漏到更早时点的可见快照。
        if not frame.available_ns.eq(frame.event_ns).all():
            fail("UNSUPPORTED_CAPABILITY", "Delayed research closes need a publication-aware provider", source=path)
        first[key.trading_day] = min(first.get(key.trading_day, result.first_ns), result.first_ns)
        last[key.trading_day] = max(last.get(key.trading_day, result.last_ns), result.last_ns)
        rows = frame.itertuples(index=False) if minute else frame.tail(1).itertuples(index=False)
        closes.extend(ObservedClose(key.symbol.lower(), key.trading_day, int(row.event_ns),
                                    Decimal(str(row.close))) for row in rows)
    return tuple(closes), first, last


def load_roll_anchor_closes(files, time_spec=None):
    spec = time_spec or BarReadSpec(required_fields=("close",), value_policy="close_strict")
    return _collect_closes(files, minute=False, time_spec=spec)[0]


def load_role_minute_closes(files, time_spec=None):
    spec = time_spec or BarReadSpec(required_fields=("close",), value_policy="close_strict")
    return _collect_closes(files, minute=True, time_spec=spec)[0]


def _contracts(row, roles, product, day):
    import re
    result = {role: str(row[ROLE_COLUMNS[role]]).strip().lower() for role in roles}
    if any(re.fullmatch(re.escape(product) + r"\d+", code, re.IGNORECASE) is None
           for code in result.values()):
        fail("INVALID_ROLE", f"{day}/{product}: invalid real contracts {result}")
    return result


def _available_ns(row, effective, timezone):
    import pandas as pd
    value = row.get("available_ns")
    if value is not None and pd.notna(value):
        try:
            decimal = Decimal(str(value))
            if not decimal.is_finite() or decimal < 0 or decimal != decimal.to_integral_value():
                raise ValueError("Not a nonnegative integer")
            number = int(decimal)
        except (ValueError, ArithmeticError):
            fail("INVALID_TIMESTAMP", "Role availability must be nonnegative integer nanoseconds")
        return max(effective, number)
    return effective


def build_role_store(assignments, closes, *, roll_policy="previous_common", max_anchor_lookback=1):
    from bomber.framework.datahub.role_prices import RolePriceStore
    return RolePriceStore(tuple(assignments), tuple(closes), missing_roll_policy=roll_policy,
                         max_anchor_lookback_trading_days=max_anchor_lookback)


def build_sector_store(role_stores, days, first_by_day, signal_role, execution_product,
                       execution_role):
    from bomber.framework.datahub.sector_roles import SectorRoleAssignment, SectorRoleStore
    records = []
    for day in days:
        effective = first_by_day[day]
        try:
            selected = {product: store.assignment_at(effective) for product, store in role_stores.items()}
            factors = {product: {signal_role: store.factor_at(effective, signal_role)[1]}
                       for product, store in role_stores.items()}
        except LookupError as exc:
            fail("REFERENCE_NOT_AVAILABLE", f"{day}: role/factor unavailable at first completed bar: {exc}")
        if any(row.trading_day != day for row in selected.values()):
            fail("REFERENCE_NOT_AVAILABLE", f"{day}: no complete effective role snapshot")
        source_days = {row.source_day for row in selected.values()}
        if len(source_days) != 1:
            fail("CONFLICTING_METADATA", f"{day}: products have different role source days")
        if execution_role not in selected[execution_product].contracts:
            fail("REFERENCE_NOT_AVAILABLE", f"Missing execution role {execution_role}")
        records.append(SectorRoleAssignment(day, source_days.pop(), effective,
            max(row.available_ns for row in selected.values()),
            {product: row.contracts for product, row in selected.items()}, factors))
    return SectorRoleStore(tuple(records))


def load_sector_research(*, bars_dir, contract_struct_path, signal_products, signal_role,
                         execution_product, execution_role, end_day=None,
                         timezone="Asia/Shanghai", bar_timestamp="end",
                         factors_path=None, factor_date_basis=None,
                         factor_availability="explicit"):
    from bomber.framework.datahub.role_prices import RoleAssignment
    products = tuple(p.strip().upper() for p in signal_products)
    if not products or len(set(products)) != len(products) or execution_product not in products:
        raise ValueError("Signal products must be unique and include the execution product")
    required_roles = {p: tuple(dict.fromkeys((signal_role, execution_role)))
                      if p == execution_product else (signal_role,) for p in products}
    roles = load_role_assignments(contract_struct_path, products, required_roles)
    index = {key: path for key, path in scan_bar_files(bars_dir, end_day=end_day).items()
             if key.symbol.rstrip("0123456789").upper() in products}
    if not index:
        raise FileNotFoundError(f"No bars for {products}: {bars_dir}")
    spec = BarReadSpec(required_fields=("close",), value_policy="close_strict",
                       timezone=timezone, timestamp_label=bar_timestamp)
    closes, first, last = _collect_closes(index, minute=False, time_spec=spec)
    days = tuple(sorted(first))
    stores = {}
    external_assignments = {}
    for product in products:
        assignments = []
        product_rows = roles.loc[roles.code.eq(product)]
        for day in days:
            prior = product_rows.loc[product_rows.source_day.lt(day)]
            if prior.empty:
                continue
            row = prior.iloc[-1]
            used_roles = required_roles[product]
            assignments.append(RoleAssignment(day, row.source_day, first[day],
                _available_ns(row, first[day], timezone), _contracts(row, used_roles, product, day),
                roles=used_roles))
        if not assignments:
            raise ValueError(f"{product}没有可用的上一交易日角色记录")
        if factors_path is not None:
            external_assignments[product] = {row.trading_day: row for row in assignments}
        else:
            stores[product] = build_role_store(assignments,
                [close for close in closes if close.instrument.rstrip("0123456789").upper() == product])
    if factors_path is not None:
        from .factors import load_cumulative_factors, build_external_sector_store
        required_factor_keys = {
            (row.source_day if factor_date_basis == "source" else day, product, role)
            for product, rows in external_assignments.items() for day, row in rows.items()
            for role in row.contracts
        }
        factors = load_cumulative_factors(factors_path, date_basis=factor_date_basis,
            availability=factor_availability, timezone=timezone,
            required_keys=required_factor_keys)
        store = build_external_sector_store(external_assignments, first,
            signal_role=signal_role, factors=factors, date_basis=factor_date_basis)
    else:
        store = build_sector_store(stores, days, first, signal_role, execution_product, execution_role)
    return LoadedSectorResearch(store, tuple((day, last[day]) for day in days), Path(bars_dir))


def load_role_research(*, bars_dir, contract_struct_path, product, end_day,
                       timezone="Asia/Shanghai", bar_timestamp="end",
                       signal_roles=("main", "secondary", "far")):
    from bomber.framework.datahub.role_prices import RoleAssignment
    product = product.strip().upper()
    if not product.isalpha():
        raise ValueError("品种代码必须是字母")
    roles = load_role_assignments(contract_struct_path, (product,), signal_roles)
    columns = [ROLE_COLUMNS[role] for role in signal_roles]
    selected = {str(code).strip().upper() for code in
                roles.loc[roles.source_day.lt(end_day), columns].to_numpy().flat}
    index = {key: path for key, path in scan_bar_files(bars_dir, end_day=end_day).items()
             if key.symbol in selected}
    if not index:
        raise FileNotFoundError(f"No {product} role bars until {end_day}: {bars_dir}")
    spec = BarReadSpec(required_fields=("close",), value_policy="close_strict",
                       timezone=timezone, timestamp_label=bar_timestamp)
    closes, first, last = _collect_closes(index, minute=True, time_spec=spec)
    assignments = []
    for day in sorted(first):
        prior = roles.loc[roles.source_day.lt(day)]
        if prior.empty:
            continue
        row = prior.iloc[-1]
        assignments.append(RoleAssignment(day, row.source_day, first[day],
            _available_ns(row, first[day], timezone), _contracts(row, signal_roles, product, day),
            roles=signal_roles))
    if not assignments:
        raise ValueError(f"{product} 没有可用的上一交易日角色记录")
    store = build_role_store(assignments, closes)
    return LoadedRoleResearch(store, tuple((row.trading_day, last[row.trading_day]) for row in assignments),
                              len(closes), len(index))
