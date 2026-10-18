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
Polymarket decentralized prediction market integration adapter.

This subpackage provides instrument providers, data and execution client configurations,
factories, constants, and credential helpers for connecting to and interacting with
the Polymarket Central Limit Order Book (CLOB) API.

For convenience, the most commonly used symbols are re-exported at the subpackage's
top level, so downstream code can simply import from ``bomber.adapters.polymarket``.

"""

from bomber.adapters.polymarket.common.constants import POLYMARKET
from bomber.adapters.polymarket.common.constants import POLYMARKET_CLIENT_ID
from bomber.adapters.polymarket.common.constants import POLYMARKET_MAX_PRECISION_MAKER
from bomber.adapters.polymarket.common.constants import POLYMARKET_MAX_PRECISION_TAKER
from bomber.adapters.polymarket.common.constants import POLYMARKET_MAX_PRICE
from bomber.adapters.polymarket.common.constants import POLYMARKET_MIN_PRICE
from bomber.adapters.polymarket.common.constants import POLYMARKET_VENUE
from bomber.adapters.polymarket.common.parsing import parse_polymarket_instrument
from bomber.adapters.polymarket.common.symbol import get_polymarket_instrument_id
from bomber.adapters.polymarket.config import PolymarketDataClientConfig
from bomber.adapters.polymarket.config import PolymarketExecClientConfig
from bomber.adapters.polymarket.factories import PolymarketLiveDataClientFactory
from bomber.adapters.polymarket.factories import PolymarketLiveExecClientFactory
from bomber.adapters.polymarket.factories import get_polymarket_http_client
from bomber.adapters.polymarket.factories import get_polymarket_instrument_provider
from bomber.adapters.polymarket.loaders import PolymarketDataLoader
from bomber.adapters.polymarket.providers import PolymarketInstrumentProvider


__all__ = [
    "POLYMARKET",
    "POLYMARKET_CLIENT_ID",
    "POLYMARKET_MAX_PRECISION_MAKER",
    "POLYMARKET_MAX_PRECISION_TAKER",
    "POLYMARKET_MAX_PRICE",
    "POLYMARKET_MIN_PRICE",
    "POLYMARKET_VENUE",
    "PolymarketDataClientConfig",
    "PolymarketDataLoader",
    "PolymarketExecClientConfig",
    "PolymarketInstrumentProvider",
    "PolymarketLiveDataClientFactory",
    "PolymarketLiveExecClientFactory",
    "get_polymarket_http_client",
    "get_polymarket_instrument_id",
    "get_polymarket_instrument_provider",
    "parse_polymarket_instrument",
]
