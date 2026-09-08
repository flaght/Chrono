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
"""
Bybit Python v2 data tester example.

The default path builds a live node and attaches the built-in Rust DataTester without
connecting to Bybit. Pass --run to start subscriptions.

"""

from __future__ import annotations

import argparse

from bomber.adapters.bybit import BybitDataClientConfig
from bomber.adapters.bybit import BybitDataClientFactory
from bomber.adapters.bybit import BybitEnvironment
from bomber.adapters.bybit import BybitProductType
from bomber.common import Environment
from bomber.live import LiveNode
from bomber.model import ClientId
from bomber.model import InstrumentId
from bomber.model import TraderId
from bomber.testkit import DataTesterConfig


BYBIT = "BYBIT"


def main() -> None:
    args = parse_args()
    instrument_id = InstrumentId.from_str(args.instrument)

    builder = LiveNode.builder(
        "BYBIT-DATA-TESTER-001",
        TraderId.from_str(args.trader_id),
        Environment.LIVE,
    ).add_data_client(
        None,
        BybitDataClientFactory(),
        BybitDataClientConfig(
            product_types=[BybitProductType.LINEAR],
            environment=BybitEnvironment.MAINNET,
        ),
    )

    node = builder.build()
    node.add_builtin_actor(
        "DataTester",
        DataTesterConfig(
            client_id=ClientId.from_str(BYBIT),
            instrument_ids=[instrument_id],
            subscribe_quotes=True,
            subscribe_trades=True,
            subscribe_mark_prices=True,
            subscribe_index_prices=True,
            subscribe_funding_rates=True,
            manage_book=True,
            log_data=True,
        ),
    )

    if args.run:
        node.run()
    else:
        print("Built Bybit data tester node. Pass --run to connect.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or run the Bybit Python v2 data tester.")
    parser.add_argument("--trader-id", default="TESTER-001")
    parser.add_argument("--instrument", default=f"BTCUSDT-LINEAR.{BYBIT}")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
