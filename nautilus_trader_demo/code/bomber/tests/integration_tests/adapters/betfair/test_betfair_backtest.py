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

from bomber.adapters.betfair.constants import BETFAIR_VENUE
from bomber.adapters.betfair.parsing.core import BetfairParser
from bomber.backtest.engine import BacktestEngine
from bomber.backtest.engine import BacktestEngineConfig
from bomber.backtest.engine import Decimal
from bomber.config import LoggingConfig
from bomber.examples.strategies.orderbook_imbalance import OrderBookImbalance
from bomber.examples.strategies.orderbook_imbalance import OrderBookImbalanceConfig
from bomber.model.currencies import GBP
from bomber.model.enums import AccountType
from bomber.model.enums import BookType
from bomber.model.enums import OmsType
from bomber.model.identifiers import ClientId
from bomber.model.identifiers import TraderId
from bomber.model.objects import Money
from tests.integration_tests.adapters.betfair.test_kit import BetfairDataProvider
from tests.integration_tests.adapters.betfair.test_kit import betting_instrument


def test_betfair_backtest():
    # Arrange
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(bypass_logging=True),
    )

    # Build the backtest engine
    engine = BacktestEngine(config=config)

    # Add a trading venue (multiple venues possible)
    engine.add_venue(
        venue=BETFAIR_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,  # Spot CASH account (not for perpetuals or futures)
        base_currency=GBP,  # Multi-currency account
        starting_balances=[Money(100_000, GBP)],
        book_type=BookType.L1_MBP,
    )

    # Add instruments
    min_notional = Money(1, GBP)
    instruments = [
        betting_instrument(
            market_id="1-166811431",
            selection_id=19248890,
            selection_handicap=None,
        ),
        betting_instrument(
            market_id="1-166811431",
            selection_id=38848248,
            selection_handicap=None,
        ),
    ]
    engine.add_instrument(instruments[0])
    engine.add_instrument(instruments[1])

    # Add data
    raw = list(BetfairDataProvider.market_updates())
    parser = BetfairParser(currency="GBP")
    updates = [upd for update in raw for upd in parser.parse(update, min_notional=min_notional)]
    engine.add_data(updates, client_id=ClientId("BETFAIR"))

    # Configure your strategy
    strategies = [
        OrderBookImbalance(
            config=OrderBookImbalanceConfig(
                instrument_id=instrument.id,
                max_trade_size=Decimal(10),
                order_id_tag=instrument.selection_id,
            ),
        )
        for instrument in instruments
    ]
    engine.add_strategies(strategies)

    # Act
    engine.run()

    # Assert
    account = engine.trader.generate_account_report(BETFAIR_VENUE)
    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    assert account.iloc[-1]["total"] == "906476.18"
    assert len(fills) == 1518
    assert len(positions) == 4
