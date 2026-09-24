"""用同一EMA策略装配CTP或Binance离线Bar的正式模拟回测。

本文件是装配层，不是策略逻辑：选择文件/Parser、交易规则Profile、
风险阈值和执行Backend；策略只接收primary_bar并输出position目标。
逐块说明见同目录README.md的「积木式架构」章节。
"""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bomber.backtest.config import BacktestEngineConfig
from bomber.model import Venue
from bomber.model.currencies import BTC
from bomber.model.identifiers import TraderId



from trader import (
    BinanceUsdtFuturesProfile,
    CtpFuturesBasicProfile,
    DataBinding,
    ExecutionRoute,
    MarketReferencePriceStore,
    MarketStreamBinding,
    NautilusMarketFeedAdapter,
    NautilusSimExecutionBackend,
    NetTargetOrderPlanner,
    PositionManager,
    PreTradeRiskManager,
    RiskLimits,
    RuntimeMode,
    SimulationExecutionClient,
    UnifiedHistoricalRuntime,
    UnifiedStrategyRunner,
)


from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import (
    BarColumns,
    BinanceKlineParser,
    BinanceMarketType,
    MappedBarParser
)

from examples.single_ema.strategies import EmaCrossConfig, EmaCrossTargetStrategy

DEFAULT_BN_BAR_PATH = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/binance_data/"
    "futures/um/klines/1m/BTCUSDT/2023-09-02.csv",
)

DEFAULT_CTP_BAR_PATH = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/cn_futures/20260728/"
    "rb2704_20260728.feather",
)


def create_instrument(instrument_name: str):
    """同时构造交易侧Profile/合约及行情侧InstrumentMeta。

    Profile决定模拟Venue的账户、保证金和手续费规则；合约描述可交易标的；
    InstrumentMeta供FileReplayFeed按正确精度解析Bar，三者不能互相替代。
    """
    if instrument_name == 'BTCUSDT':
        profile = BinanceUsdtFuturesProfile(
            starting_balance=Decimal("100000"),
            leverage=Decimal("10"),
            maintenance_margin_rate=Decimal("0.005"),
            maker_fee=Decimal("0.0002"),
            taker_fee=Decimal("0.0005"),
        )

        instrument = profile.make_instrument(
            instrument_name,
            price_precision=2,
            size_precision=6,
            price_increment=Decimal("0.01"),
            size_increment=Decimal("0.000001"),
            base_currency=BTC,
            min_quantity=Decimal("0.000001"),
            min_notional=Decimal("5"),
        )

        instrument_id = instrument.id
        instrument_meta = InstrumentMeta(
            instrument_id=instrument_id,
            price_precision=2,
            size_precision=6,
            price_increment=Decimal("0.01"),
            multiplier=Decimal(1),
            currency="USDT",
            exchange="BINANCE",
        )
    elif instrument_name == "rb2704":
        profile = CtpFuturesBasicProfile(
            starting_balance=Decimal("100000"),
            venue=Venue("SHFE"))

        def utc_ns(value: str) -> int:
            return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000_000_000)

        instrument = profile.make_instrument(
            instrument_name,
            underlying="rb",
            price_precision=0,
            price_increment=Decimal("1"),
            multiplier=Decimal("10"),
            activation_ns=utc_ns("2025-01-01"),
            expiration_ns=utc_ns("2027-05-01"),
            margin_init=Decimal("0.10"),
            margin_maint=Decimal("0.08"),
        )

        instrument_id = instrument.id
        instrument_meta = InstrumentMeta(
            instrument_id=instrument_id,
            price_precision=0,
            size_precision=0,
            price_increment=Decimal(1),
            multiplier=Decimal(10),
            currency="CNY",
            exchange="SHFE",
        )
    else:
        raise ValueError(f"不支持的回测合约: {instrument_name}")
    return profile, instrument, instrument_meta


def create_parser(instrument_name: str):
    """只适配文件格式，统一输出Bar；不承载EMA或下单规则。"""
    if instrument_name == 'BTCUSDT':
        #         MappedBarParser(
        #         columns=BarColumns(
        #             symbol="symbol",
        #             exchange="exchange",
        #             timestamp="bar_close_time",
        #             open="o",
        #             high="h",
        #             low="l",
        #             close="c",
        #             volume="vol",
        #         ),
        #         bar_spec="1-MINUTE",
        #         timezone="UTC",
        #     ),
        # )
        parser = BinanceKlineParser(
            symbol="BTCUSDT",
            market_type=BinanceMarketType.FUTURES,
            interval="1m",
            include_factors=False,
        )
    else:
        parser = MappedBarParser(
            columns=BarColumns(
                symbol="symbol",
                exchange="exchange",
                timestamp="datetime",
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                value="value",
                open_interest="open_interest",
                vwap="vwap",
            ),
            bar_spec="1-MINUTE",
            timezone="Asia/Shanghai",
            exchange_aliases={"XSGE": "SHFE"},
        )
    return parser


def create_data_bindings(instrument_name: str, instrument_id: str):
    """把策略逻辑输入primary_bar绑定到具体Feed、合约和1分钟Bar。"""
    if instrument_name == 'BTCUSDT':
        data_bindings = (
            DataBinding(
                "primary_bar",
                f"{instrument_name}-replay",
                instrument_id,
                DataType.BAR,
                "1-MINUTE",
            ),
        )
    else:
        data_bindings = (
            DataBinding(
                "primary_bar",
                f"{instrument_name}-replay",
                instrument_id,
                DataType.BAR,
                "1-MINUTE"
            ),
        )
    return data_bindings


def create_risk_limits(instrument_name: str):
    """按标的设置前置风控阈值；参考价另由行情Observer实时维护。"""
    if instrument_name == 'BTCUSDT':
        RISK_LIMITS = RiskLimits(
            max_order_quantity=Decimal("0.01"),          # 单笔最多 0.01 BTC
            max_abs_position=Decimal("0.02"),            # 绝对仓位最多 0.02 BTC
            max_order_notional=Decimal("2000"),          # 单笔名义金额最多 2,000 USDT
            max_abs_position_notional=Decimal("3000"),   # 持仓名义金额最多 3,000 USDT
            max_market_age_ns=180 * 1_000_000_000,       # 参考行情最多过期 180 秒
            contract_multiplier=Decimal("1"),            # 当前 BTCUSDT 永续合约定义的乘数
        )
    else:
        RISK_LIMITS = RiskLimits(
            max_order_quantity=Decimal(2),
            max_abs_position=Decimal(1),
            max_order_notional=Decimal("50000"),
            max_abs_position_notional=Decimal("50000"),
            max_market_age_ns=120 * 1_000_000_000,
            contract_multiplier=Decimal(10),
        )
    return RISK_LIMITS


def run_case(instrument_name: str,
             path: Path,
             quantity: Decimal,  # 仓位 0,001  1
             fast: int = 3, slow: int = 5):
    """只装配一轮回测；真正执行从调用者的runtime.run()开始。"""

    # 输入积木：Reader由文件后缀选择，Parser按来源解释列，Feed发布标准Bar。
    profile, instrument, instrument_meta = create_instrument(
        instrument_name=instrument_name)
    feed = FileReplayFeed("EMA_REPLAY")
    instrument_id = instrument.id
    parser = create_parser(instrument_name=instrument_name)
    feed.register_instrument(instrument_meta)
    if path.suffix.lower() == '.feather':
        feed.add_bar_feather(path, parser)
    elif path.suffix.lower() == '.csv':
        feed.add_bar_csv(path, parser)

    # 执行积木：Backend持有Nautilus模拟引擎；Profile和合约决定撮合规则。
    backend = NautilusSimExecutionBackend(
        f"{instrument_name}-sim",
        BacktestEngineConfig(trader_id=TraderId(
            "BACKTESTER-001"), run_analysis=True),
    )
    risk_limit = create_risk_limits(instrument_name=instrument_name)

    backend.add_profile(profile)
    backend.add_instrument(instrument)

    # PositionManager保存真实/模拟账户仓位及在途量；价格存储只保存风控参考价。
    positions = PositionManager()
    prices = MarketReferencePriceStore()

    data_bindings = create_data_bindings(
        instrument_name=instrument_name, instrument_id=instrument_id)

    # 与DataBinding不同：MarketStreamBinding让同一份Bar推进模拟时钟与旧订单撮合。
    market_bindings = (
        MarketStreamBinding(instrument_id, DataType.BAR, "1-MINUTE"),
    )

    # 目标仓位 -> Planner订单意图 -> Risk检查 -> Backend模拟订单与成交回报。
    client = SimulationExecutionClient(
        backend.backend_id,
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=PreTradeRiskManager(client_id=backend.backend_id,
                                         position_manager=positions, price_store=prices,
                                         instrument_limits={instrument_id: risk_limit})
    )
    # 业务策略只认识primary_bar和position，不直接依赖文件、Profile或Backend。
    target_strategy = EmaCrossTargetStrategy(
        "ema-offline",
        EmaCrossConfig(
            fast_period=fast,
            slow_period=slow,
            long_quantity=quantity,  # 快 EMA 高于或等于慢 EMA 时，输出的目标仓位
            short_quantity=-quantity,
            skip_single_price=True  # True 当前bar为单价格K线时跳过, False 不跳过
        ),
    )

    # Adapter只转发行情到Backend；生命周期由UnifiedHistoricalRuntime统一管理。
    market_adapter = NautilusMarketFeedAdapter(
        f"{instrument_name}-clock",
        feed,
        backend,
        market_bindings,
        manage_lifecycle=False,
    )

    # Runner管理Feed、策略、目标存储、组合净额和执行路由。
    runner = UnifiedStrategyRunner(
        RuntimeMode.HISTORICAL, position_manager=positions)
    
    # 旁路观察者先更新参考价；策略发单时RiskLimits才能检查名义金额和行情时效。
    runner.add_market_observer(prices)
    # Feed ID必须与DataBinding中的ID相同，否则策略收不到Bar。
    runner.add_data_feed(f"{instrument_name}-replay", feed)
    runner.add_execution_client(client)

    # DataBinding是「行情从哪来」，ExecutionRoute是「目标交给谁、交易什么」。
    runner.add_strategy(
        target_strategy,
        data_bindings=data_bindings,
        execution_routes=(
            ExecutionRoute("position", backend.backend_id, instrument_id),
        ),
    )

    
    # 先挂模拟时钟回调、再挂策略回调：当前Bar先撮合旧订单，再形成新信号。
    runtime = UnifiedHistoricalRuntime(
        f"{instrument_name}-runtime",
        runner,
        market_adapter,
    )

    return runtime, target_strategy, client, backend, instrument_id


def main(instrument_name):

    if instrument_name == 'rb2704':
        default_bar_path = DEFAULT_CTP_BAR_PATH
        quantity=Decimal("1")

    elif instrument_name == 'BTCUSDT':
        default_bar_path = DEFAULT_BN_BAR_PATH
        quantity = Decimal("0.001")

    runtime, target_strategy, client, backend, instrument_id = run_case(
        instrument_name=instrument_name,
        path=default_bar_path,
        quantity=quantity,
    )
    try:
        # run_case仅装配；这里才逐条回放、驱动策略和模拟撮合。
        result = runtime.run()
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        print(
            f"[{instrument_id}] loaded={result.replay_summary.total:,} "
            f"bars_used={target_strategy.bars_used:,} "
            f"last_target={target_strategy.last_target} "
            f"orders={len(orders)} fills={len(fills)}",
        )
        if target_strategy.last_target is None or orders.empty or fills.empty:
            raise AssertionError("EMA回测没有形成完整目标、订单和成交")
        if client.report_errors:
            raise AssertionError(f"回测执行回报异常: {client.report_errors}")
    finally:
        runtime.stop()


if __name__ == "__main__":
    main(instrument_name='rb2704')
    main(instrument_name='BTCUSDT')
