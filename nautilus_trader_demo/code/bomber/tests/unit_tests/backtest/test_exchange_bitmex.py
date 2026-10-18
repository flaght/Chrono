# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

from decimal import Decimal

from bomber.backtest.engine import SimulatedExchange
from bomber.backtest.execution_client import BacktestExecClient
from bomber.backtest.models import FillModel
from bomber.backtest.models import LatencyModel
from bomber.backtest.models import MakerTakerFeeModel
from bomber.common.component import MessageBus
from bomber.common.component import TestClock
from bomber.data.engine import DataEngine
from bomber.execution.engine import ExecutionEngine
from bomber.model.currencies import BTC
from bomber.model.data import QuoteTick
from bomber.model.enums import AccountType
from bomber.model.enums import LiquiditySide
from bomber.model.enums import OmsType
from bomber.model.enums import OrderSide
from bomber.model.identifiers import Venue
from bomber.model.objects import Money
from bomber.model.objects import Price
from bomber.model.objects import Quantity
from bomber.portfolio.portfolio import Portfolio
from bomber.risk.engine import RiskEngine
from bomber.test_kit.mocks.strategies import MockStrategy
from bomber.test_kit.providers import TestInstrumentProvider
from bomber.test_kit.stubs.component import TestComponentStubs
from bomber.test_kit.stubs.data import TestDataStubs
from bomber.test_kit.stubs.identifiers import TestIdStubs


XBTUSD_BITMEX = TestInstrumentProvider.xbtusd_bitmex()


class TestBitmexExchange:
    """
    Various tests which are more specific to market making with maker rebates.
    """

    def setup(self):
        # Fixture Setup
        self.strategies = [MockStrategy(TestDataStubs.bartype_btcusdt_binance_100tick_last())]

        self.clock = TestClock()
        self.trader_id = TestIdStubs.trader_id()

        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
        )

        self.cache = TestComponentStubs.cache()

        self.portfolio = Portfolio(
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        self.data_engine = DataEngine(
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        self.exec_engine = ExecutionEngine(
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        self.risk_engine = RiskEngine(
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        self.exchange = SimulatedExchange(
            venue=Venue("BITMEX"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=BTC,
            starting_balances=[Money(20, BTC)],
            default_leverage=Decimal(50),
            leverages={},
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            modules=[],
            fill_model=FillModel(),
            fee_model=MakerTakerFeeModel(),
            clock=self.clock,
            latency_model=LatencyModel(0),
        )
        self.exchange.add_instrument(XBTUSD_BITMEX)

        self.exec_client = BacktestExecClient(
            exchange=self.exchange,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Wire up components
        self.exec_engine.register_client(self.exec_client)
        self.exchange.register_client(self.exec_client)

        self.cache.add_instrument(XBTUSD_BITMEX)

        self.strategy = MockStrategy(bar_type=TestDataStubs.bartype_btcusdt_binance_100tick_last())
        self.strategy.register(
            trader_id=self.trader_id,
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        self.exchange.reset()
        self.data_engine.start()
        self.exec_engine.start()
        self.strategy.start()

    def test_commission_maker_taker_order(self):
        # Arrange
        # Prepare market
        quote1 = QuoteTick(
            instrument_id=XBTUSD_BITMEX.id,
            bid_price=Price.from_str("11493.0"),
            ask_price=Price.from_str("11493.5"),
            bid_size=Quantity.from_int(1_500_000),
            ask_size=Quantity.from_int(1_500_000),
            ts_event=0,
            ts_init=0,
        )

        self.data_engine.process(quote1)
        self.exchange.process_quote_tick(quote1)

        order_market = self.strategy.order_factory.market(
            XBTUSD_BITMEX.id,
            OrderSide.BUY,
            Quantity.from_int(100_000),
        )

        order_limit = self.strategy.order_factory.limit(
            XBTUSD_BITMEX.id,
            OrderSide.BUY,
            Quantity.from_int(100_000),
            Price.from_str("11492.5"),
        )

        # Act
        self.strategy.submit_order(order_market)
        self.exchange.process(0)
        self.strategy.submit_order(order_limit)
        self.exchange.process(0)

        quote2 = QuoteTick(
            instrument_id=XBTUSD_BITMEX.id,
            bid_price=Price.from_str("11491.0"),
            ask_price=Price.from_str("11491.5"),
            bid_size=Quantity.from_int(1_500_000),
            ask_size=Quantity.from_int(1_500_000),
            ts_event=0,
            ts_init=0,
        )

        self.exchange.process_quote_tick(quote2)  # Fill the limit order
        self.portfolio.update_quote_tick(quote2)

        # Assert
        assert order_limit.avg_px == 11492.5
        assert self.strategy.store[2].liquidity_side == LiquiditySide.TAKER
        assert self.strategy.store[7].liquidity_side == LiquiditySide.MAKER
        assert self.strategy.store[2].commission == Money(0.00652543, BTC)
        assert self.strategy.store[7].commission == Money(-0.00217533, BTC)
