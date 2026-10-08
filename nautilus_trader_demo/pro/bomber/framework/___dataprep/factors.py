"""读取外部提供的累计因子，不自行推导替代值。"""
from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from .catalog import symbol
from .contracts import InputIssue
from .session import current_session, fail, read_feather, record_issue


def load_cumulative_factors(path, *, date_basis, availability="explicit",
                            timezone="Asia/Shanghai", required_keys=None):
    """返回映射：(日期, 品种, 角色) -> (合约, 累计因子, 可用时间 available_ns)。

    长表字段：trade_date/code/symbol/role/cumulative_factor/available_ns。
    旧格式中没有 role 的 pcr_cumfactor 仅表示主力因子，不推断其他角色
    或发布时间；调用方必须明确声明 date_basis。
    """
    import pandas as pd
    if date_basis not in {"source", "trading"}:
        fail("INVALID_FACTOR_POLICY", "Declare factor date_basis as source or trading")
    if availability not in {"explicit", "source-day-end", "aligned"}:
        fail("INVALID_FACTOR_POLICY", "Unknown factor availability policy")
    if availability == "source-day-end" and date_basis != "source":
        fail("INVALID_FACTOR_POLICY", "source-day-end requires source-date factors")
    if availability == "aligned" and date_basis != "trading":
        fail("INVALID_FACTOR_POLICY", "aligned requires trading-date factors")
    frame = read_feather(path)
    if not frame.columns.is_unique or frame.empty:
        fail("INVALID_FACTOR", "Factor table must be nonempty with unique columns", source=path)
    value_column = "cumulative_factor" if "cumulative_factor" in frame else "pcr_cumfactor"
    required = {"trade_date", "code", "symbol", value_column}
    if availability == "explicit":
        required.add("available_ns")
    if required - set(frame):
        fail("MISSING_FIELD", f"Factor table missing {sorted(required - set(frame))}", source=path)
    if value_column == "cumulative_factor" and "role" not in frame:
        fail("MISSING_FIELD", "Generic cumulative_factor requires an explicit role", source=path)
    selected_keys = None if required_keys is None else set(required_keys)
    selected_pairs = None if selected_keys is None else {(p, r) for _, p, r in selected_keys}
    result = {}
    for line, row in enumerate(frame.to_dict("records"), 1):
        try:
            product = str(row["code"]).strip().upper()
            role = str(row.get("role", "main")).strip().lower()
            # 只校验当前场景请求的数据中的无效值。
            # 缺失的请求数据由存储构造器拒绝。
            if selected_pairs is not None and (product, role) not in selected_pairs:
                continue
            stamp = pd.Timestamp(row["trade_date"])
            if pd.isna(stamp):
                raise ValueError("Empty factor date")
            day = stamp.date()
            if selected_keys is not None and (day, product, role) not in selected_keys:
                continue
            contract = symbol(row["symbol"])
            value = Decimal(str(row[value_column]))
            if role not in {"main", "secondary", "near", "far"}:
                raise ValueError("Unknown factor role")
            if not product.isalpha() or not contract.startswith(product) or not contract[len(product):].isdigit():
                raise ValueError("Factor symbol must be a real contract of the declared product")
            if not value.is_finite() or value <= 0:
                raise ValueError("Cumulative factor must be positive and finite")
            if availability == "explicit" or availability == "aligned" and "available_ns" in row:
                raw = Decimal(str(row["available_ns"]))
                if not raw.is_finite() or raw < 0 or raw != raw.to_integral_value():
                    raise ValueError("available_ns must be nonnegative integer nanoseconds")
                available_ns = int(raw)
            elif availability == "source-day-end":
                # 调用方显式选择此假设：资料在本地来源日结束前发布。
                end = datetime.combine(day + timedelta(days=1), time(), ZoneInfo(timezone))
                available_ns = pd.Timestamp(end).value
            else:
                # 上游约定：已对齐的因子在对应交易日可用。
                # 零仅为内部哨兵值；快照可用时间仍采用
                #该角色首个生效 Bar 的时间，不表示资料在纪元起点发布。
                available_ns = 0
        except (ValueError, TypeError, InvalidOperation, OverflowError) as exc:
            detail = (f"{exc}; row={line}, trade_date={row.get('trade_date')!r}, "
                      f"code={row.get('code')!r}, symbol={row.get('symbol')!r}, "
                      f"role={row.get('role', 'main')!r}, {value_column}={row.get(value_column)!r}")
            fail("INVALID_FACTOR", detail, source=path, row=line,
                 symbol=str(row.get("symbol")))
        key = (day, product, role)
        record = (contract.lower(), value, available_ns)
        if key in result and result[key] != record:
            fail("CONFLICTING_METADATA", f"Conflicting factor {key}", source=path, row=line)
        result[key] = record
    session = current_session()
    if session is not None:
        session.coverage.assumptions += (f"external_factor_date_basis:{date_basis}",
            f"external_factor_availability:{availability}")
        if availability == "aligned":
            session.coverage.assumptions += (
                "upstream_aligned_factors_available_at_first_effective_bar_unless_available_ns_is_provided",)
    return result


def build_external_sector_store(assignments, first_by_day, *, signal_role,
                                factors, date_basis):
    from bomber.framework.datahub.sector_roles import SectorRoleAssignment, SectorRoleStore
    records = []
    days = sorted(first_by_day)
    for day in days:
        effective = first_by_day[day]
        selected = {product: rows[day] for product, rows in assignments.items() if day in rows}
        if len(selected) != len(assignments):
            fail("REFERENCE_NOT_AVAILABLE", f"{day}: missing effective role assignment")
        source_days = {row.source_day for row in selected.values()}
        if len(source_days) != 1:
            fail("CONFLICTING_METADATA", f"{day}: products have different role source days")
        cumulative = {}
        contracts = {product: dict(row.contracts) for product, row in selected.items()}
        available = effective
        for product, row in selected.items():
            factor_day = row.source_day if date_basis == "source" else day
            key = (factor_day, product, signal_role)
            if key not in factors:
                fail("MISSING_FACTOR", f"Missing supplied cumulative factor {key}")
            _, value, _ = factors[key]
            available = max(available, row.available_ns)
            for role, original in row.contracts.items():
                role_key = (factor_day, product, role)
                if role_key not in factors:
                    # 仅用于执行的角色不要求研究因子；若已提供因子，
                    # 仍优先采用因子表中的合约映射。
                    continue
                contract, _, publication = factors[role_key]
                available = max(available, publication)
                if contract != original.lower():
                    record_issue(InputIssue("FACTOR_CONTRACT_OVERRIDE",
                        f"{day}/{product}/{role}: role contract {original} overridden by factor symbol {contract}",
                        severity="WARNING", symbol=contract, trading_day=day))
                contracts[product][role] = contract
            cumulative[product] = {signal_role: value}
        if available > effective:
            fail("REFERENCE_NOT_AVAILABLE", f"{day}: supplied factor/role unavailable at first completed bar")
        records.append(SectorRoleAssignment(day, source_days.pop(), effective, available,
            contracts, cumulative))
    return SectorRoleStore(tuple(records))
