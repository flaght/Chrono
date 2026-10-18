"""联合入口的Binance DEMO积木工厂；构造不连接网络。"""

from bomber.framework.trader.execution.builders import ExecutionComponents
from bomber.framework.trader.execution.planner import NetTargetOrderPlanner
from bomber.framework.trader.execution.risk import PreTradeRiskManager
from .backend import NautilusLiveExecutionBackend
from .controlled import ControlledLiveExecutionClient
from .driver import NautilusTradingNodeDriver
from .binance_account import BinanceNativeAccountBinding, BinanceHttpAccountReader
from .binance_orders import BinanceNativeOpenOrdersBinding, BinanceHttpActiveOrderReader


def build_binance_joint_execution(*, instrument_id, account_id, positions, prices,
                                 risk_limits, api_key, api_secret):
    # 延迟加载原生SDK；离线联合框架测试无需建立TradingNode。
    from bomber.adapters.binance import (BINANCE, BinanceAccountType,
        BinanceDataClientConfig, BinanceExecClientConfig, BinanceInstrumentProviderConfig,
        BinanceLiveDataClientFactory, BinanceLiveExecClientFactory)
    from bomber.adapters.binance.common.enums import BinanceEnvironment
    from bomber.config import LiveExecEngineConfig, LoggingConfig, TradingNodeConfig
    from bomber.live.node import TradingNode
    from bomber.model.identifiers import TraderId

    provider = BinanceInstrumentProviderConfig(load_all=True)
    config = BinanceExecClientConfig(account_type=BinanceAccountType.USDT_FUTURES,
        environment=BinanceEnvironment.DEMO, instrument_provider=provider,
        # 原生配置为PositiveInt|None；None让订单RetryManager使用0次重试。
        api_key=api_key, api_secret=api_secret, max_retries=None)
    node = TradingNode(config=TradingNodeConfig(trader_id=TraderId("JOINT-DEMO-001"),
        logging=LoggingConfig(log_level="INFO"),
        exec_engine=LiveExecEngineConfig(reconciliation=True, graceful_shutdown_on_exception=True),
        data_clients={BINANCE: BinanceDataClientConfig(account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment.DEMO, instrument_provider=provider,
            api_key=api_key, api_secret=api_secret)}, exec_clients={BINANCE: config},
        timeout_connection=30.0, timeout_reconciliation=30.0, timeout_portfolio=30.0,
        timeout_disconnection=10.0, timeout_post_stop=5.0))
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    orders_binding, account_binding = BinanceNativeOpenOrdersBinding(), BinanceNativeAccountBinding()

    class CapturingFactory(BinanceLiveExecClientFactory):
        @staticmethod
        def create(loop, name, config, msgbus, cache, clock):
            native = BinanceLiveExecClientFactory.create(loop, name, config, msgbus, cache, clock)
            orders_binding.capture(native)
            account_binding.capture(native)
            return native

    node.add_exec_client_factory(BINANCE, CapturingFactory)
    client_id = "joint-binance-demo"
    reader = BinanceHttpAccountReader(client_id, account_id, node.get_event_loop, account_binding)
    orders = BinanceHttpActiveOrderReader(client_id, account_id, node.get_event_loop,
        orders_binding.query_open_orders, timeout_seconds=15)
    driver = NautilusTradingNodeDriver(client_id, node, reconcile_callback=reader.positions,
        account_state_callback=reader.account, active_orders_callback=orders.read,
        ready_callback=lambda: bool(node.portfolio.initialized)
            and node.portfolio.account(BINANCE) is not None
            and node.cache.instrument(instrument_id) is not None, startup_timeout=60)
    backend = NautilusLiveExecutionBackend(client_id, driver)
    client = ControlledLiveExecutionClient(client_id, NetTargetOrderPlanner(positions), backend,
        positions, PreTradeRiskManager(client_id, positions, prices,
            instrument_limits={instrument_id: risk_limits}), account_id=account_id,
        demo_environment_check=lambda: config.environment is BinanceEnvironment.DEMO and node.is_running(),
        max_request_wall_age_ns=10_000_000_000)
    return ExecutionComponents(client, positions, prices, backend=backend, driver=driver)
