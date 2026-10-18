"""期权Vega策略的共同装配；不打开数据源或选择执行渠道。"""
from bomber.framework.market.basic.base import DataType
from bomber.framework.trader.assembly import StrategyBindings, assemble_strategy
from bomber.framework.trader.contracts import DataBinding, ExecutionRoute
from bomber.framework.trader.execution.risk import RiskLimits
from .strategy import OptionVegaStrategy


def risk_limits(config, contracts, instruments, multipliers, max_notional):
    return {instrument: RiskLimits(
        max_order_quantity=(config.max_option_lots if key in contracts else config.max_future_lots) * 2,
        max_abs_position=config.max_option_lots if key in contracts else config.max_future_lots,
        max_order_notional=max_notional, max_abs_position_notional=max_notional,
        max_market_age_ns=config.max_market_age_seconds * 1_000_000_000,
        contract_multiplier=multipliers[instrument]) for key, instrument in instruments.items()}


def assemble_strategy_components(runner, execution, *, config, contracts, futures, instruments,
        index_key, index_id, trading_days, options_by_day, feeds, feed_by_key,
        final_day=None, strategy_id="option-vega"):
    all_ids = {**instruments, index_key: index_id}
    if set(feed_by_key) != set(all_ids):
        raise ValueError("行情绑定须覆盖期权、期货及指数")
    strategy = OptionVegaStrategy(strategy_id, config, contracts, futures, instruments,
        index_key, trading_days, final_day, options_by_day)
    bindings = StrategyBindings(
        data=tuple(DataBinding(key, feed_by_key[key], item, DataType.BAR, "1-MINUTE")
            for key, item in all_ids.items()),
        execution=tuple(ExecutionRoute(str(item), execution.client.client_id, item)
            for item in instruments.values()))
    assemble_strategy(runner, strategy, feeds=feeds, execution=execution, bindings=bindings)
    return strategy
