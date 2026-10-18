#!/usr/bin/env python3
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

from strategy import DemoStrategy
from strategy import DemoStrategyConfig

from examples.utils.data_provider import prepare_demo_data_eurusd_futures_1min
from bomber.backtest.engine import BacktestEngine
from bomber.config import BacktestEngineConfig
from bomber.config import LoggingConfig
from bomber.core.bomber_pyo3 import BarType
from bomber.model import Bar
from bomber.model import TraderId
from bomber.model.currencies import USD
from bomber.model.enums import AccountType
from bomber.model.enums import OmsType
from bomber.model.identifiers import Venue
from bomber.model.instruments.base import Instrument
from bomber.model.objects import Money


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # 1. Configure and create backtest engine
    # ----------------------------------------------------------------------------------

    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTEST_TRADER-001"),
        logging=LoggingConfig(
            log_level="DEBUG",  # Enable debug logging
        ),
    )
    engine = BacktestEngine(config=engine_config)

    # ----------------------------------------------------------------------------------
    # 2. Prepare market data
    # ----------------------------------------------------------------------------------

    prepared_data: dict = prepare_demo_data_eurusd_futures_1min()
    venue_name: str = prepared_data["venue_name"]
    eurusd_instrument: Instrument = prepared_data["instrument"]
    eurusd_1min_bartype: BarType = prepared_data["bar_type"]
    eurusd_1min_bars_list: list[Bar] = prepared_data["bars_list"]

    # ----------------------------------------------------------------------------------
    # 3. Configure trading environment
    # ----------------------------------------------------------------------------------

    # Set up the trading venue with a margin account
    engine.add_venue(
        venue=Venue(venue_name),
        oms_type=OmsType.NETTING,  # Order Management System type
        account_type=AccountType.MARGIN,  # Type of trading account
        starting_balances=[Money(1_000_000, USD)],  # Initial account balance
        base_currency=USD,  # Base currency for account
        default_leverage=Decimal(1),  # No leverage used for account
    )

    # Add instrument and market data to the engine
    engine.add_instrument(eurusd_instrument)
    engine.add_data(eurusd_1min_bars_list)

    # ----------------------------------------------------------------------------------
    # 4. Configure and run strategy
    # ----------------------------------------------------------------------------------

    # Create and register the portfolio strategy with configuration
    strategy_config = DemoStrategyConfig(
        bar_type=eurusd_1min_bartype,
        instrument=eurusd_instrument,
    )
    strategy = DemoStrategy(config=strategy_config)
    engine.add_strategy(strategy)

    # Execute the backtest
    engine.run()

    # Clean up resources
    engine.dispose()
