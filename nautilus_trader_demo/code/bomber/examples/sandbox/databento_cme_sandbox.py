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
Sandbox for Databento live data and CME simulated execution.
"""

from bomber.adapters.databento import DATABENTO
from bomber.adapters.databento import DatabentoDataClientConfig
from bomber.adapters.databento import DatabentoLiveDataClientFactory
from bomber.adapters.sandbox.config import SandboxExecutionClientConfig
from bomber.adapters.sandbox.factory import SandboxLiveExecClientFactory
from bomber.config import InstrumentProviderConfig
from bomber.config import LiveExecEngineConfig
from bomber.config import LoggingConfig
from bomber.config import TradingNodeConfig
from bomber.examples.strategies.simpler_quoter import SimpleQuoterStrategy
from bomber.examples.strategies.simpler_quoter import SimpleQuoterStrategyConfig
from bomber.live.node import TradingNode
from bomber.model.identifiers import InstrumentId
from bomber.model.identifiers import TraderId


# Specify instrument to be traded
instrument_id = InstrumentId.from_str("ESZ6.XCME")

instrument_provider = InstrumentProviderConfig(load_ids=frozenset([instrument_id]))

# Configure the trading node:
# For correct subscription operation, you must specify all instruments to be immediately
# subscribed for as part of the data client configuration.
config_data = DatabentoDataClientConfig(
    http_gateway=None,
    instrument_provider=instrument_provider,
    use_exchange_as_venue=True,
    mbo_subscriptions_delay=10.0,
    instrument_ids=[instrument_id],
    parent_symbols={"GLBX.MDP3": {"ES.FUT"}},
)

config_exec = SandboxExecutionClientConfig(
    venue="XCME",
    base_currency="USD",
    starting_balances=["1_000_000 USD"],
    instrument_provider=instrument_provider,
)

config_node = TradingNodeConfig(
    trader_id=TraderId("SANDBOX-001"),
    logging=LoggingConfig(log_level="INFO", use_pyo3=True),
    exec_engine=LiveExecEngineConfig(reconciliation=False),
    data_clients={
        DATABENTO: config_data,
    },
    exec_clients={
        "XCME": config_exec,
    },
    timeout_connection=30.0,
    timeout_reconciliation=5.0,
    timeout_portfolio=5.0,
    timeout_disconnection=10.0,
    timeout_post_stop=2.0,
)

# Instantiate the node with a configuration
node = TradingNode(config=config_node)

# Configure and initialize the quoter strategy
config_quoter = SimpleQuoterStrategyConfig(
    instrument_id=instrument_id,
    tob_offset_ticks=0,
    log_data=False,
)
quoter = SimpleQuoterStrategy(config=config_quoter)

node.trader.add_strategy(quoter)

# Register required client factories with the node
node.add_data_client_factory(DATABENTO, DatabentoLiveDataClientFactory)
node.add_exec_client_factory("XCME", SandboxLiveExecClientFactory)
node.build()


# Stop and dispose of the node with SIGINT/CTRL+C
if __name__ == "__main__":
    try:
        node.run()
    finally:
        node.dispose()
