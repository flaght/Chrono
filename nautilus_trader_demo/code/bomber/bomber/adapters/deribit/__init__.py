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
The Deribit adapter provides integration with the Deribit cryptocurrency derivatives
exchange, supporting live market data ingest and order execution.

This adapter supports:
- Market data streaming via WebSocket (trades, order book, quotes).
- Order execution via WebSocket (market, limit, stop market, stop limit).
- Instrument definitions for futures, options, spot, and combo instruments.
- Multiple currencies (BTC, ETH, USDC, USDT, EURR).

"""

from bomber.adapters.deribit.config import DeribitDataClientConfig
from bomber.adapters.deribit.config import DeribitExecClientConfig
from bomber.adapters.deribit.constants import DERIBIT
from bomber.adapters.deribit.constants import DERIBIT_CLIENT_ID
from bomber.adapters.deribit.constants import DERIBIT_VENUE
from bomber.adapters.deribit.data import DeribitDataClient
from bomber.adapters.deribit.execution import DeribitExecutionClient
from bomber.adapters.deribit.factories import DeribitLiveDataClientFactory
from bomber.adapters.deribit.factories import DeribitLiveExecClientFactory
from bomber.adapters.deribit.factories import get_cached_deribit_http_client
from bomber.adapters.deribit.factories import get_cached_deribit_instrument_provider
from bomber.adapters.deribit.providers import DeribitInstrumentProvider
from bomber.core.bomber_pyo3 import DeribitCurrency
from bomber.core.bomber_pyo3 import DeribitHttpClient
from bomber.core.bomber_pyo3 import DeribitProductType
from bomber.core.bomber_pyo3 import DeribitUpdateInterval
from bomber.core.bomber_pyo3 import DeribitWebSocketClient


__all__ = [
    "DERIBIT",
    "DERIBIT_CLIENT_ID",
    "DERIBIT_VENUE",
    "DeribitCurrency",
    "DeribitDataClient",
    "DeribitDataClientConfig",
    "DeribitExecClientConfig",
    "DeribitExecutionClient",
    "DeribitHttpClient",
    "DeribitInstrumentProvider",
    "DeribitLiveDataClientFactory",
    "DeribitLiveExecClientFactory",
    "DeribitProductType",
    "DeribitUpdateInterval",
    "DeribitWebSocketClient",
    "get_cached_deribit_http_client",
    "get_cached_deribit_instrument_provider",
]
