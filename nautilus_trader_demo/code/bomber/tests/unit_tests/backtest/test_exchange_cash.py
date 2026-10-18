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
from bomber.config import ExecEngineConfig
from bomber.config import RiskEngineConfig
from bomber.data.engine import DataEngine
from bomber.execution.engine import ExecutionEngine
from bomber.model.currencies import USD
from bomber.model.data import BarType
from bomber.model.enums import AccountType
from bomber.model.enums import OmsType
from bomber.model.enums import OrderSide
from bomber.model.enums import OrderStatus
from bomber.model.identifiers import Venue
from bomber.model.objects import Money
from bomber.model.objects import Quantity
from bomber.portfolio.portfolio import Portfolio
from bomber.risk.engine import RiskEngine
from bomber.test_kit.mocks.strategies import MockStrategy
from bomber.test_kit.providers import TestInstrumentProvider
from bomber.test_kit.stubs.component import TestComponentStubs
from bomber.test_kit.stubs.data import TestDataStubs
from bomber.test_kit.stubs.identifiers import TestIdStubs


_AAPL_XNAS = TestInstrumentProvider.equity()


class TestSimulatedExchangeCashAccount:
    def setup(self) -> None:
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
            config=ExecEngineConfig(debug=True),
        )

        self.risk_engine = RiskEngine(
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
            config=RiskEngineConfig(debug=True),
        )

        self.exchange = SimulatedExchange(
            venue=Venue("XNAS"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            base_currency=USD,
            starting_balances=[Money(1_000_000, USD)],
            default_leverage=Decimal(0),
            leverages={},
            modules=[],
            fill_model=FillModel(),
            fee_model=MakerTakerFeeModel(),
            portfolio=self.portfolio,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
            latency_model=LatencyModel(0),
        )
        self.exchange.add_instrument(_AAPL_XNAS)

        self.exec_client = BacktestExecClient(
            exchange=self.exchange,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )

        # Wire up components
        self.exec_engine.register_client(self.exec_client)
        self.exchange.register_client(self.exec_client)

        self.cache.add_instrument(_AAPL_XNAS)

        # Create mock strategy
        self.strategy = MockStrategy(bar_type=BarType.from_str("AAPL.XNAS-1-MINUTE-BID-INTERNAL"))
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

    def test_repr(self) -> None:
        # Arrange, Act, Assert
        assert (
            repr(self.exchange) == "SimulatedExchange(id=XNAS, oms_type=NETTING, account_type=CASH)"
        )

    def test_equity_short_selling_will_reject(self) -> None:
        # Arrange: Prepare market
        quote1 = TestDataStubs.quote_tick(
            instrument=_AAPL_XNAS,
            bid_price=100.00,
            ask_price=101.00,
        )
        self.data_engine.process(quote1)
        self.exchange.process_quote_tick(quote1)

        # Act
        order1 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            OrderSide.BUY,
            Quantity.from_int(100),
        )
        self.strategy.submit_order(order1)
        self.exchange.process(0)

        order2 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            OrderSide.SELL,
            Quantity.from_int(110),
        )
        self.strategy.submit_order(order2)
        self.exchange.process(0)

        position_id = self.cache.positions_open()[0].id  # Generated by exchange
        order3 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            OrderSide.SELL,
            Quantity.from_int(100),
        )
        self.strategy.submit_order(order3, position_id=position_id)
        self.exchange.process(0)

        order4 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            OrderSide.SELL,
            Quantity.from_int(100),
        )
        self.strategy.submit_order(order4)
        self.exchange.process(0)

        # Assert
        assert order1.status == OrderStatus.FILLED
        assert order2.status == OrderStatus.REJECTED
        assert order3.status == OrderStatus.FILLED
        assert order4.status == OrderStatus.REJECTED
        assert self.exchange.get_account().balance_total(USD) == Money(999_900, USD)

    def test_equity_selling_will_not_reject_with_cash_netting(self) -> None:
        # Arrange: Prepare market
        quote1 = TestDataStubs.quote_tick(
            instrument=_AAPL_XNAS,
            bid_price=100.00,
            ask_price=101.00,
        )
        self.data_engine.process(quote1)
        self.exchange.process_quote_tick(quote1)

        # Act
        order1 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            OrderSide.BUY,
            Quantity.from_int(200),
        )
        self.strategy.submit_order(order1)
        self.exchange.process(0)

        order2 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            OrderSide.SELL,
            Quantity.from_int(100),
        )
        self.strategy.submit_order(order2)
        self.exchange.process(0)

        order3 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            OrderSide.SELL,
            Quantity.from_int(100),
        )
        self.strategy.submit_order(order3)
        self.exchange.process(0)

        # Assert
        assert order1.status == OrderStatus.FILLED
        assert order2.status == OrderStatus.FILLED
        assert order3.status == OrderStatus.FILLED
        assert self.exchange.get_account().balance_total(USD) == Money(999_800, USD)

    @pytest.mark.parametrize(
        ("entry_side", "expected_usd"),
        [
            [OrderSide.BUY, Money(979_800.00, USD)],
        ],
    )
    def test_equity_order_fills_for_entry(
        self,
        entry_side: OrderSide,
        expected_usd: Money,
    ) -> None:
        # Arrange: Prepare market
        quote1 = TestDataStubs.quote_tick(
            instrument=_AAPL_XNAS,
            bid_price=100.00,
            ask_price=101.00,
        )
        self.data_engine.process(quote1)
        self.exchange.process_quote_tick(quote1)

        order1 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            entry_side,
            Quantity.from_int(100),
        )

        order2 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            entry_side,
            Quantity.from_int(100),
        )

        # Act
        self.strategy.submit_order(order1)
        self.exchange.process(0)

        self.strategy.submit_order(order2)
        self.exchange.process(0)

        # Assert
        assert order1.status == OrderStatus.FILLED
        assert order2.status == OrderStatus.FILLED
        assert self.exchange.get_account().balance_total(USD) == expected_usd

    @pytest.mark.parametrize(
        ("entry_side", "exit_side", "expected_usd"),
        [
            [OrderSide.BUY, OrderSide.SELL, Money(984_650.00, USD)],
        ],
    )
    def test_equity_order_fills_with_partial_exit(
        self,
        entry_side: OrderSide,
        exit_side: OrderSide,
        expected_usd: Money,
    ) -> None:
        # Arrange: Prepare market
        quote1 = TestDataStubs.quote_tick(
            instrument=_AAPL_XNAS,
            bid_price=100.00,
            ask_price=101.00,
        )
        self.data_engine.process(quote1)
        self.exchange.process_quote_tick(quote1)

        order1 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            entry_side,
            Quantity.from_int(100),
        )

        order2 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            entry_side,
            Quantity.from_int(100),
        )

        quote2 = TestDataStubs.quote_tick(
            instrument=_AAPL_XNAS,
            bid_price=101.00,
            ask_price=102.00,
        )
        self.data_engine.process(quote2)
        self.exchange.process_quote_tick(quote2)

        order3 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            exit_side,
            Quantity.from_int(50),
        )

        # Act
        self.strategy.submit_order(order1)
        self.exchange.process(0)

        position_id = self.cache.positions_open()[0].id  # Generated by exchange

        self.strategy.submit_order(order2)
        self.exchange.process(0)
        self.strategy.submit_order(order3, position_id=position_id)
        self.exchange.process(0)

        # Assert
        assert order1.status == OrderStatus.FILLED
        assert order2.status == OrderStatus.FILLED
        assert order3.status == OrderStatus.FILLED
        assert self.exchange.get_account().balance_total(USD) == expected_usd

    @pytest.mark.parametrize(
        ("entry_side", "exit_side", "expected_usd"),
        [
            [OrderSide.BUY, OrderSide.SELL, Money(999_800.00, USD)],
        ],
    )
    def test_equity_order_multiple_entry_fills(
        self,
        entry_side: OrderSide,
        exit_side: OrderSide,
        expected_usd: Money,
    ) -> None:
        # Arrange: Prepare market
        quote1 = TestDataStubs.quote_tick(
            instrument=_AAPL_XNAS,
            bid_price=100.00,
            ask_price=101.00,
        )
        self.data_engine.process(quote1)
        self.exchange.process_quote_tick(quote1)

        order1 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            entry_side,
            Quantity.from_int(100),
        )

        order2 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            exit_side,
            Quantity.from_int(100),
        )

        order3 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            entry_side,
            Quantity.from_int(100),
        )

        order4 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            exit_side,
            Quantity.from_int(100),
        )

        # Act
        self.strategy.submit_order(order1)
        self.exchange.process(0)

        position_id = self.cache.positions_open()[0].id  # Generated by exchange

        self.strategy.submit_order(order2, position_id=position_id)
        self.exchange.process(0)

        self.strategy.submit_order(order3)
        self.exchange.process(0)
        self.strategy.submit_order(order4, position_id=position_id)
        self.exchange.process(0)

        # Assert
        assert order1.status == OrderStatus.FILLED
        assert order2.status == OrderStatus.FILLED
        assert order3.status == OrderStatus.FILLED
        assert order4.status == OrderStatus.FILLED
        assert not self.cache.positions_open()
        assert self.exchange.get_account().balance_total(USD) == expected_usd
        assert len(self.exchange.get_account().events) == 5

    def test_book_depth_can_fill(
        self,
    ) -> None:
        # Arrange: Prepare market
        depth = TestDataStubs.order_book_depth10(instrument_id=_AAPL_XNAS.id)
        self.data_engine.process(depth)
        self.exchange.process_order_book_depth10(depth)

        order1 = self.strategy.order_factory.market(
            _AAPL_XNAS.id,
            OrderSide.BUY,
            Quantity.from_int(100),
        )

        # Act
        self.strategy.submit_order(order1)
        self.exchange.process(0)

        # Assert
        assert order1.status == OrderStatus.FILLED
