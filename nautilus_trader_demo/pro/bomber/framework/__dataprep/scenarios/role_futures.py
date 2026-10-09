"""角色研究与真实合约执行数据的通用组装，不选择策略交易角色。"""
from ..references import load_role_research, load_sector_research
from ..contracts import BarReadSpec, InputContext
from .fixed_contracts import plan_fixed_contracts, prepare_fixed_contracts


def plan_role_futures(required, optional=(), **kwargs):
    return plan_fixed_contracts(required, optional, **kwargs)


def prepare_role_research(*, bars_dir, contract_struct_path, products, signal_role="main",
                         execution_product=None, execution_role="main", end_day=None,
                         minute_roles=(), timezone="Asia/Shanghai", bar_timestamp="end",
                         factors_path=None, factor_date_basis=None,
                         factor_availability="explicit"):
    products = tuple(products)
    if not products:
        raise ValueError("Role research requires at least one product")
    if minute_roles:
        if factors_path is not None:
            raise ValueError("External factors are currently supported by the daily sector scene only")
        if len(products) != 1 or end_day is None:
            raise ValueError("Minute-role research requires one product and end_day")
        return load_role_research(bars_dir=bars_dir, contract_struct_path=contract_struct_path,
            product=products[0], end_day=end_day, timezone=timezone, bar_timestamp=bar_timestamp,
            signal_roles=tuple(minute_roles))
    return load_sector_research(bars_dir=bars_dir, contract_struct_path=contract_struct_path,
        signal_products=tuple(products), signal_role=signal_role,
        execution_product=execution_product or products[0], execution_role=execution_role,
        end_day=end_day, timezone=timezone, bar_timestamp=bar_timestamp,
        factors_path=factors_path, factor_date_basis=factor_date_basis,
        factor_availability=factor_availability)


def prepare_role_futures(*, plan, context=None, **research_args):
    research = prepare_role_research(**research_args)
    original = context or InputContext()
    references = dict(original.references)
    references.update(role_research=research.store, day_end_ns=research.day_end_ns)
    context = InputContext(original.instruments, references, original.calendar)
    spec = BarReadSpec(timezone=research_args.get("timezone", "Asia/Shanghai"),
        timestamp_label=research_args.get("bar_timestamp", "end"))
    return prepare_fixed_contracts(research_args["bars_dir"], plan, spec=spec, context=context)
