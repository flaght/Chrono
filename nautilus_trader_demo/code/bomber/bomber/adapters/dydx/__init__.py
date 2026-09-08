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
DYdX v4 cryptocurrency exchange adapter (Rust-backed implementation).

The v4 adapter uses Rust-backed HTTP, WebSocket, and gRPC clients for:
- Native Cosmos SDK transaction signing via Rust
- Direct validator node communication
- Improved performance and reliability
- Real-time market data streaming

Usage:
    from bomber.adapters.dydx import DydxDataClientConfig
    from bomber.adapters.dydx import DydxExecClientConfig
    from bomber.adapters.dydx import DydxLiveDataClientFactory
    from bomber.adapters.dydx import DydxLiveExecClientFactory
    from bomber.adapters.dydx import DydxNetwork

"""

from bomber.adapters.dydx.config import DydxDataClientConfig
from bomber.adapters.dydx.config import DydxExecClientConfig
from bomber.adapters.dydx.constants import DYDX
from bomber.adapters.dydx.constants import DYDX_CLIENT_ID
from bomber.adapters.dydx.constants import DYDX_VENUE
from bomber.adapters.dydx.data import DydxDataClient
from bomber.adapters.dydx.execution import DydxExecutionClient
from bomber.adapters.dydx.factories import DydxLiveDataClientFactory
from bomber.adapters.dydx.factories import DydxLiveExecClientFactory
from bomber.adapters.dydx.providers import DydxInstrumentProvider
from bomber.core.bomber_pyo3 import DydxNetwork


__all__ = [
    "DYDX",
    "DYDX_CLIENT_ID",
    "DYDX_VENUE",
    "DydxDataClient",
    "DydxDataClientConfig",
    "DydxExecClientConfig",
    "DydxExecutionClient",
    "DydxInstrumentProvider",
    "DydxLiveDataClientFactory",
    "DydxLiveExecClientFactory",
    "DydxNetwork",
]
