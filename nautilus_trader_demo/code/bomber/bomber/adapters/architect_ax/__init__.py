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
AX Exchange integration adapter.

This subpackage provides an instrument provider, data and execution clients,
configurations, data types and constants for connecting to and interacting with
the AX Exchange API.

For convenience, the most commonly used symbols are re-exported at the
subpackage's top level, so downstream code can simply import from
``bomber.adapters.architect_ax``.

"""

from bomber.adapters.architect_ax.config import AxDataClientConfig
from bomber.adapters.architect_ax.config import AxExecClientConfig
from bomber.adapters.architect_ax.constants import AX
from bomber.adapters.architect_ax.constants import AX_CLIENT_ID
from bomber.adapters.architect_ax.constants import AX_VENUE
from bomber.adapters.architect_ax.factories import AxLiveDataClientFactory
from bomber.adapters.architect_ax.factories import AxLiveExecClientFactory
from bomber.adapters.architect_ax.factories import get_cached_ax_http_client
from bomber.adapters.architect_ax.factories import get_cached_ax_instrument_provider
from bomber.adapters.architect_ax.providers import AxInstrumentProvider
from bomber.core.bomber_pyo3 import AxEnvironment
from bomber.core.bomber_pyo3 import AxMarketDataLevel


__all__ = [
    "AX",
    "AX_CLIENT_ID",
    "AX_VENUE",
    "AxDataClientConfig",
    "AxEnvironment",
    "AxExecClientConfig",
    "AxInstrumentProvider",
    "AxLiveDataClientFactory",
    "AxLiveExecClientFactory",
    "AxMarketDataLevel",
    "get_cached_ax_http_client",
    "get_cached_ax_instrument_provider",
]
