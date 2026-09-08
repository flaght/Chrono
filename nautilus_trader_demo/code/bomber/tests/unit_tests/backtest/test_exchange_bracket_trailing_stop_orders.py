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

import pytest

from bomber.backtest.engine import SimulatedExchange
from bomber.backtest.execution_client import BacktestExecClient
from bomber.backtest.models import FillModel
from bomber.backtest.models import LatencyModel
from bomber.backtest.models import MakerTakerFeeModel
from bomber.common.component import MessageBus
from bomber.common.component import TestClock
from bomber.data.engine import DataEngine
from bomber.execution.emulator import OrderEmulator
from bomber.execution.engine import ExecutionEngine
from bomber.model.currencies import ETH
from bomber.model.currencies import USDT
from bomber.model.enums import AccountType
from bomber.model.enums import OmsType
from bomber.model.enums import OrderSide
from bomber.model.enums import OrderStatus
from bomber.model.enums import OrderType
from bomber.model.enums import TrailingOffsetType
from bomber.model.enums import TriggerType
from bomber.model.identifiers import Venue
from bomber.model.objects import Money
from bomber.portfolio.portfolio import Portfolio
from bomber.risk.engine import RiskEngine
from bomber.test_kit.mocks.strategies import MockStrategy
from bomber.test_kit.providers import TestInstrumentProvider
from bomber.test_kit.stubs.component import TestComponentStubs
from bomber.test_kit.stubs.data import TestDataStubs
from bomber.test_kit.stubs.identifiers import TestIdStubs


BINANCE = Venue("BINANCE")
ETHUSDT_PERP_BINANCE = TestInstrumentProvider.ethusdt_perp_binance()


class TestSimulatedExchangeEmulatedContingencyOrders:
    def setup(self):
        # Fixture Setup
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
            clock=self.clock,
            cache=self.cache,
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

        self.emulator = OrderEmulator(
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        self.exchange = SimulatedExchange(
            venue=BINANCE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=None,  # Multi-asset wallet
            starting_balances=[Money(200, ETH), Money(1_000_000, USDT)],
            default_leverage=Decimal(10),
            leverages={},
            modules=[],
            fill_model=FillModel(),
            fee_model=MakerTakerFeeModel(),
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
            latency_model=LatencyModel(0),
            reject_stop_orders=False,
        )
        self.exchange.add_instrument(ETHUSDT_PERP_BINANCE)

        self.exec_client = BacktestExecClient(
            exchange=self.exchange,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Wire up components
        self.exec_engine.register_client(self.exec_client)
        self.exchange.register_client(self.exec_client)

        self.cache.add_instrument(ETHUSDT_PERP_BINANCE)

        # Create mock strategy
        self.strategy = MockStrategy(bar_type=TestDataStubs.bartype_usdjpy_1min_bid())
        self.strategy.register(
            trader_id=self.trader_id,
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Start components
        self.exchange.reset()
        self.data_engine.start()
        self.exec_engine.start()
        self.strategy.start()

    @pytest.mark.parametrize(
        "emulation_trigger",
        [
            TriggerType.NO_TRIGGER,
            TriggerType.BID_ASK,
        ],
    )
    def test_bracket_accepts_trailing_stop_market_order_tp(
        self,
        emulation_trigger: TriggerType,
    ) -> None:
        # Arrange: Prepare market
        tick1 = TestDataStubs.quote_tick(
            instrument=ETHUSDT_PERP_BINANCE,
            bid_price=3100.00,
            ask_price=3100.00,
            ts_event=0,
            ts_init=0,
        )
        self.data_engine.process(tick1)
        self.exchange.process_quote_tick(tick1)

        bracket = self.strategy.order_factory.bracket(
            instrument_id=ETHUSDT_PERP_BINANCE.id,
            order_side=OrderSide.BUY,
            quantity=ETHUSDT_PERP_BINANCE.make_qty(10.000),
            emulation_trigger=emulation_trigger,
            entry_order_type=OrderType.MARKET,
            tp_order_type=OrderType.TRAILING_STOP_MARKET,
            tp_activation_price=ETHUSDT_PERP_BINANCE.make_price(3200.00),
            tp_trigger_price=None,
            tp_trigger_type=TriggerType.BID_ASK,
            tp_trailing_offset=ETHUSDT_PERP_BINANCE.make_price(30),
            tp_trailing_offset_type=TrailingOffsetType.PRICE,
            sl_order_type=OrderType.STOP_MARKET,
            sl_trigger_price=ETHUSDT_PERP_BINANCE.make_price(3000.00),
        )

        # Act
        self.strategy.submit_order_list(bracket)
        self.exchange.process(0)

        # Assert
        entry_order = bracket.orders[0]
        sl_order = bracket.orders[1]
        tp_order = bracket.orders[2]
        assert entry_order.status == OrderStatus.FILLED
        assert (
            sl_order.status == OrderStatus.EMULATED
            if sl_order.is_emulated
            else OrderStatus.ACCEPTED
        )
        assert (
            tp_order.status == OrderStatus.EMULATED
            if tp_order.is_emulated
            else OrderStatus.ACCEPTED
        )

        # Act
        tick2 = TestDataStubs.quote_tick(
            instrument=ETHUSDT_PERP_BINANCE,
            bid_price=3250.00,
            ask_price=3250.00,
        )
        self.data_engine.process(tick2)
        self.exchange.process_quote_tick(tick2)
        self.emulator.on_quote_tick(tick2)
        self.exchange.process(0)

        # Assert
        tp_order = self.cache.order(bracket.orders[2].client_order_id)
        assert tp_order.is_activated

        # Act
        tick3 = TestDataStubs.quote_tick(
            instrument=ETHUSDT_PERP_BINANCE,
            bid_price=3210.00,
            ask_price=3210.00,
        )
        self.data_engine.process(tick3)
        self.exchange.process_quote_tick(tick3)
        self.emulator.on_quote_tick(tick3)
        self.exchange.process(0)

        # Assert
        sl_order = self.cache.order(bracket.orders[1].client_order_id)
        tp_order = self.cache.order(bracket.orders[2].client_order_id)
        assert sl_order.status == OrderStatus.CANCELED
        assert tp_order.status == OrderStatus.FILLED

    @pytest.mark.parametrize(
        "emulation_trigger",
        [
            TriggerType.NO_TRIGGER,
            TriggerType.BID_ASK,
        ],
    )
    def test_bracket_accepts_trailing_stop_market_order_sl(
        self,
        emulation_trigger: TriggerType,
    ) -> None:
        # Arrange: Prepare market
        tick1 = TestDataStubs.quote_tick(
            instrument=ETHUSDT_PERP_BINANCE,
            bid_price=3100.00,
            ask_price=3100.00,
            ts_event=0,
            ts_init=0,
        )
        self.data_engine.process(tick1)
        self.exchange.process_quote_tick(tick1)

        bracket = self.strategy.order_factory.bracket(
            instrument_id=ETHUSDT_PERP_BINANCE.id,
            order_side=OrderSide.BUY,
            quantity=ETHUSDT_PERP_BINANCE.make_qty(10.000),
            emulation_trigger=emulation_trigger,
            entry_order_type=OrderType.MARKET,
            tp_price=ETHUSDT_PERP_BINANCE.make_price(3200.00),
            sl_order_type=OrderType.TRAILING_STOP_MARKET,
            sl_activation_price=None,  # activated at the current market price
            sl_trigger_type=TriggerType.BID_ASK,
            sl_trailing_offset=ETHUSDT_PERP_BINANCE.make_price(100.00),
            sl_trailing_offset_type=TrailingOffsetType.PRICE,
        )

        # Act
        self.strategy.submit_order_list(bracket)
        self.exchange.process(0)

        # Assert
        entry_order = bracket.orders[0]
        sl_order = bracket.orders[1]
        tp_order = bracket.orders[2]
        assert entry_order.status == OrderStatus.FILLED
        assert (
            sl_order.status == OrderStatus.EMULATED
            if sl_order.is_emulated
            else OrderStatus.ACCEPTED
        )
        assert sl_order.is_activated
        assert sl_order.activation_price == ETHUSDT_PERP_BINANCE.make_price(3100.00)
        assert sl_order.trigger_price == ETHUSDT_PERP_BINANCE.make_price(3000.00)
        assert (
            tp_order.status == OrderStatus.EMULATED
            if tp_order.is_emulated
            else OrderStatus.ACCEPTED
        )

        # Act
        tick2 = TestDataStubs.quote_tick(
            instrument=ETHUSDT_PERP_BINANCE,
            bid_price=3150.00,
            ask_price=3150.00,
        )
        self.data_engine.process(tick2)
        self.exchange.process_quote_tick(tick2)
        self.emulator.on_quote_tick(tick2)
        self.exchange.process(0)

        # Assert
        # sl_order's trigger price should reflect the changed market,
        # and its status should not be changed.
        sl_order = self.cache.order(bracket.orders[1].client_order_id)
        assert sl_order.trigger_price == ETHUSDT_PERP_BINANCE.make_price(3050.00)
        assert (
            sl_order.status == OrderStatus.EMULATED
            if sl_order.is_emulated
            else OrderStatus.ACCEPTED
        )

        # Act
        tick3 = TestDataStubs.quote_tick(
            instrument=ETHUSDT_PERP_BINANCE,
            bid_price=3050.00,
            ask_price=3050.00,
        )
        self.data_engine.process(tick3)
        self.exchange.process_quote_tick(tick3)
        self.emulator.on_quote_tick(tick3)
        self.exchange.process(0)

        # Assert
        sl_order = self.cache.order(bracket.orders[1].client_order_id)
        tp_order = self.cache.order(bracket.orders[2].client_order_id)
        assert sl_order.status == OrderStatus.FILLED
        assert tp_order.status == OrderStatus.CANCELED
