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

from bomber.adapters.polymarket import POLYMARKET
from bomber.adapters.polymarket import PolymarketDataClientConfig
from bomber.adapters.polymarket import PolymarketLiveDataClientFactory
from bomber.adapters.polymarket import get_polymarket_instrument_id
from bomber.adapters.polymarket.providers import PolymarketInstrumentProviderConfig
from bomber.config import LiveExecEngineConfig
from bomber.config import LoggingConfig
from bomber.config import TradingNodeConfig
from bomber.live.node import TradingNode
from bomber.model.identifiers import TraderId
from bomber.test_kit.strategies.tester_data import DataTester
from bomber.test_kit.strategies.tester_data import DataTesterConfig


# For correct subscription operation, you must specify all instruments to be immediately
# subscribed for as part of the data client configuration

# To find active markets run `python bomber/adapters/polymarket/scripts/active_markets.py`

# Slug: gta-vi-released-before-june-2026
# Active: True
# Condition ID: 0xcccb7e7613a087c132b69cbf3a02bece3fdcb824c1da54ae79acc8d4a562d902
# Token IDs: 8441400852834915183759801017793514978104486628517653995211751018945988243154,
# 109289569086508934142323222102974769075074494425163878721602922903101062859033
# Link: https://polymarket.com/event/gta-vi-released-before-june-2026
condition_id = "0xcccb7e7613a087c132b69cbf3a02bece3fdcb824c1da54ae79acc8d4a562d902"
token_id = "8441400852834915183759801017793514978104486628517653995211751018945988243154"

instrument_ids = [
    get_polymarket_instrument_id(condition_id, token_id),
]

filters = {
    # "next_cursor": "MTE3MDA=",
    "is_active": True,
}

load_ids = [str(x) for x in instrument_ids]
instrument_provider_config = PolymarketInstrumentProviderConfig(load_ids=frozenset(load_ids))
# instrument_provider_config = PolymarketInstrumentProviderConfig(load_all=True, filters=filters)

# Configure the trading node
config_node = TradingNodeConfig(
    trader_id=TraderId("TESTER-001"),
    logging=LoggingConfig(log_level="INFO", use_pyo3=True),
    exec_engine=LiveExecEngineConfig(
        reconciliation=False,  # Not applicable
    ),
    data_clients={
        POLYMARKET: PolymarketDataClientConfig(
            signature_type=2,  # Browser wallet proxy (Polymarket UI); requires funder address
            instrument_config=instrument_provider_config,
            compute_effective_deltas=True,
        ),
    },
    timeout_connection=20.0,
    timeout_disconnection=10.0,
    timeout_post_stop=1.0,
)

# Instantiate the node with a configuration
node = TradingNode(config=config_node)

# Configure and initialize the tester
config_tester = DataTesterConfig(
    instrument_ids=instrument_ids,
    # subscribe_book_deltas=True,
    subscribe_book_at_interval=True,
    # subscribe_quotes=True,
    # subscribe_trades=True,
    book_interval_ms=10,
    can_unsubscribe=False,  # Polymarket does not support unsubscribing from ws streams
)
tester = DataTester(config=config_tester)

node.trader.add_actor(tester)

# Register your client factories with the node (can take user-defined factories)
node.add_data_client_factory(POLYMARKET, PolymarketLiveDataClientFactory)
node.build()


# Stop and dispose of the node with SIGINT/CTRL+C
if __name__ == "__main__":
    try:
        node.run()
    finally:
        node.dispose()
