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

import time
from decimal import Decimal

import pandas as pd

from bomber import TEST_DATA_DIR
from bomber.adapters.databento import DatabentoDataLoader
from bomber.backtest.config import BacktestEngineConfig
from bomber.backtest.engine import BacktestEngine
from bomber.config import LoggingConfig
from bomber.config import RiskEngineConfig
from bomber.examples.strategies.ema_cross_long_only import EMACrossLongOnly
from bomber.examples.strategies.ema_cross_long_only import EMACrossLongOnlyConfig
from bomber.model.currencies import USD
from bomber.model.data import BarType
from bomber.model.enums import AccountType
from bomber.model.enums import OmsType
from bomber.model.identifiers import TraderId
from bomber.model.identifiers import Venue
from bomber.model.objects import Money
from bomber.test_kit.providers import TestInstrumentProvider


if __name__ == "__main__":
    # Configure backtest engine
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level="INFO"),
        risk_engine=RiskEngineConfig(bypass=True),
    )

    # Build the backtest engine
    engine = BacktestEngine(config=config)

    # Add a trading venue (multiple venues possible)
    NASDAQ = Venue("XNAS")  # <-- ISO 10383 MIC
    engine.add_venue(
        venue=NASDAQ,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(1_000_000.0, USD)],
    )

    # Add instruments
    AAPL_XNAS = TestInstrumentProvider.equity(symbol="AAPL", venue="XNAS")
    engine.add_instrument(AAPL_XNAS)

    # Add data
    loader = DatabentoDataLoader()

    filenames = [
        "aapl-xnas-ohlcv-1s-2023.dbn.zst",  # <-- Longer load and run time / more accurate execution
        "aapl-xnas-ohlcv-1m-2023.dbn.zst",
    ]

    for filename in filenames:
        bars = loader.from_dbn_file(
            path=TEST_DATA_DIR / "databento" / "temp" / filename,
            instrument_id=AAPL_XNAS.id,
        )
        engine.add_data(bars)

    # Configure your strategy
    strategy_config = EMACrossLongOnlyConfig(
        instrument_id=AAPL_XNAS.id,
        bar_type=BarType.from_str(f"{AAPL_XNAS.id}-1-MINUTE-LAST-EXTERNAL"),
        trade_size=Decimal(100),
        fast_ema_period=10,
        slow_ema_period=20,
    )

    # Instantiate and add your strategy
    strategy = EMACrossLongOnly(config=strategy_config)
    engine.add_strategy(strategy=strategy)

    time.sleep(0.1)
    input("Press Enter to continue...")

    # Run the engine (from start to end of data)
    engine.run()

    # Optionally view reports
    with pd.option_context(
        "display.max_rows",
        100,
        "display.max_columns",
        None,
        "display.width",
        300,
    ):
        print(engine.trader.generate_account_report(NASDAQ))
        print(engine.trader.generate_order_fills_report())
        print(engine.trader.generate_positions_report())

    # For repeated backtest runs make sure to reset the engine
    engine.reset()

    # Good practice to dispose of the object
    engine.dispose()
