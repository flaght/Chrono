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


import pytest

from bomber.adapters.bybit.config import BybitDataClientConfig
from bomber.adapters.bybit.config import BybitExecClientConfig
from bomber.adapters.bybit.config import _resolve_environment
from bomber.adapters.bybit.data import BybitDataClient
from bomber.adapters.bybit.execution import BybitExecutionClient
from bomber.adapters.bybit.factories import BybitLiveDataClientFactory
from bomber.adapters.bybit.factories import BybitLiveExecClientFactory
from bomber.cache.cache import Cache
from bomber.common.component import LiveClock
from bomber.common.component import MessageBus
from bomber.core import bomber_pyo3
from bomber.core.bomber_pyo3 import BybitEnvironment
from bomber.test_kit.mocks.cache_database import MockCacheDatabase
from bomber.test_kit.stubs.identifiers import TestIdStubs


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (None, BybitEnvironment.MAINNET),
        (BybitEnvironment.DEMO, BybitEnvironment.DEMO),
        (BybitEnvironment.TESTNET, BybitEnvironment.TESTNET),
    ],
)
def test_resolve_environment(environment, expected):
    assert _resolve_environment(environment) == expected


class TestBybitFactories:
    @pytest.fixture(autouse=True)
    def setup(self, request):
        self.loop = request.getfixturevalue("event_loop")
        self.clock = LiveClock()

        self.trader_id = TestIdStubs.trader_id()
        self.strategy_id = TestIdStubs.strategy_id()
        self.account_id = TestIdStubs.account_id()

        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
        )

        self.cache_db = MockCacheDatabase()
        self.cache = Cache(database=self.cache_db)

        return

    @pytest.mark.parametrize(
        ("environment", "expected"),
        [
            [bomber_pyo3.BybitEnvironment.MAINNET, "https://api.bybit.com"],
            [bomber_pyo3.BybitEnvironment.TESTNET, "https://api-testnet.bybit.com"],
            [bomber_pyo3.BybitEnvironment.DEMO, "https://api-demo.bybit.com"],
        ],
    )
    def test_get_http_base_url(self, environment, expected):
        base_url = bomber_pyo3.get_bybit_http_base_url(environment)
        assert base_url == expected

    @pytest.mark.parametrize(
        ("product_type", "environment", "expected"),
        [
            [
                bomber_pyo3.BybitProductType.SPOT,
                bomber_pyo3.BybitEnvironment.MAINNET,
                "wss://stream.bybit.com/v5/public/spot",
            ],
            [
                bomber_pyo3.BybitProductType.SPOT,
                bomber_pyo3.BybitEnvironment.TESTNET,
                "wss://stream-testnet.bybit.com/v5/public/spot",
            ],
            [
                bomber_pyo3.BybitProductType.SPOT,
                bomber_pyo3.BybitEnvironment.DEMO,
                "wss://stream.bybit.com/v5/public/spot",
            ],
            [
                bomber_pyo3.BybitProductType.LINEAR,
                bomber_pyo3.BybitEnvironment.MAINNET,
                "wss://stream.bybit.com/v5/public/linear",
            ],
            [
                bomber_pyo3.BybitProductType.LINEAR,
                bomber_pyo3.BybitEnvironment.TESTNET,
                "wss://stream-testnet.bybit.com/v5/public/linear",
            ],
            [
                bomber_pyo3.BybitProductType.LINEAR,
                bomber_pyo3.BybitEnvironment.DEMO,
                "wss://stream.bybit.com/v5/public/linear",
            ],
            [
                bomber_pyo3.BybitProductType.INVERSE,
                bomber_pyo3.BybitEnvironment.MAINNET,
                "wss://stream.bybit.com/v5/public/inverse",
            ],
            [
                bomber_pyo3.BybitProductType.INVERSE,
                bomber_pyo3.BybitEnvironment.TESTNET,
                "wss://stream-testnet.bybit.com/v5/public/inverse",
            ],
            [
                bomber_pyo3.BybitProductType.INVERSE,
                bomber_pyo3.BybitEnvironment.DEMO,
                "wss://stream.bybit.com/v5/public/inverse",
            ],
        ],
    )
    def test_get_ws_base_url(self, product_type, environment, expected):
        base_url = bomber_pyo3.get_bybit_ws_url_public(product_type, environment)
        assert base_url == expected

    def test_create_bybit_live_data_client(self):
        data_client = BybitLiveDataClientFactory.create(
            loop=self.loop,
            name="BYBIT",
            config=BybitDataClientConfig(
                api_key="SOME_BYBIT_API_KEY",
                api_secret="SOME_BYBIT_API_SECRET",
                product_types=[bomber_pyo3.BybitProductType.LINEAR],
            ),
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        assert isinstance(data_client, BybitDataClient)

    def test_create_bybit_live_exec_client(self):
        data_client = BybitLiveExecClientFactory.create(
            loop=self.loop,
            name="BYBIT",
            config=BybitExecClientConfig(
                api_key="SOME_BYBIT_API_KEY",
                api_secret="SOME_BYBIT_API_SECRET",
                product_types=[bomber_pyo3.BybitProductType.LINEAR],
            ),
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        assert isinstance(data_client, BybitExecutionClient)
