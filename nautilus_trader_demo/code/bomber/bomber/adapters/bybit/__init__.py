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
Bybit cryptocurreny exchange integration adapter.

This subpackage provides an instrument provider, data and execution clients,
configurations, data types and constants for connecting to and interacting with
Bybit's API.

For convenience, the most commonly used symbols are re-exported at the
subpackage's top level, so downstream code can simply import from
``bomber.adapters.bybit``.

"""

from bomber.adapters.bybit.config import BybitDataClientConfig
from bomber.adapters.bybit.config import BybitExecClientConfig
from bomber.adapters.bybit.constants import BYBIT
from bomber.adapters.bybit.constants import BYBIT_CLIENT_ID
from bomber.adapters.bybit.constants import BYBIT_VENUE
from bomber.adapters.bybit.factories import BybitLiveDataClientFactory
from bomber.adapters.bybit.factories import BybitLiveExecClientFactory
from bomber.adapters.bybit.factories import get_cached_bybit_http_client
from bomber.adapters.bybit.factories import get_cached_bybit_instrument_provider
from bomber.adapters.bybit.loaders import BybitOrderBookDeltaDataLoader
from bomber.adapters.bybit.providers import BybitInstrumentProvider
from bomber.core.bomber_pyo3 import BybitEnvironment
from bomber.core.bomber_pyo3 import BybitMarginAction
from bomber.core.bomber_pyo3 import BybitMarginBorrowResult
from bomber.core.bomber_pyo3 import BybitMarginRepayResult
from bomber.core.bomber_pyo3 import BybitMarginStatusResult
from bomber.core.bomber_pyo3 import BybitPositionIdx
from bomber.core.bomber_pyo3 import BybitPositionMode
from bomber.core.bomber_pyo3 import BybitProductType
from bomber.core.bomber_pyo3 import BybitTickerData


__all__ = [
    "BYBIT",
    "BYBIT_CLIENT_ID",
    "BYBIT_VENUE",
    "BybitDataClientConfig",
    "BybitEnvironment",
    "BybitExecClientConfig",
    "BybitInstrumentProvider",
    "BybitLiveDataClientFactory",
    "BybitLiveExecClientFactory",
    "BybitMarginAction",
    "BybitMarginBorrowResult",
    "BybitMarginRepayResult",
    "BybitMarginStatusResult",
    "BybitOrderBookDeltaDataLoader",
    "BybitPositionIdx",
    "BybitPositionMode",
    "BybitProductType",
    "BybitTickerData",
    "get_cached_bybit_http_client",
    "get_cached_bybit_instrument_provider",
]
