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

import asyncio
from functools import lru_cache

from bomber.adapters.databento.config import DatabentoDataClientConfig
from bomber.adapters.databento.constants import PUBLISHERS_FILEPATH
from bomber.adapters.databento.data import DatabentoDataClient
from bomber.adapters.databento.loaders import DatabentoDataLoader
from bomber.adapters.databento.providers import DatabentoInstrumentProvider
from bomber.adapters.env import get_env_key
from bomber.cache.cache import Cache
from bomber.common.component import LiveClock
from bomber.common.component import MessageBus
from bomber.config import InstrumentProviderConfig
from bomber.core import bomber_pyo3
from bomber.live.factories import LiveDataClientFactory


@lru_cache(1)
def get_cached_databento_http_client(
    key: str | None = None,
    gateway: str | None = None,
    use_exchange_as_venue: bool = True,
) -> bomber_pyo3.DatabentoHistoricalClient:
    """
    Cache and return a Databento historical HTTP client with the given key and gateway.

    If a cached client with matching parameters already exists, the cached client will be returned.

    Parameters
    ----------
    key : str, optional
        The Databento API secret key for the client.
    gateway : str, optional
        The Databento historical HTTP client gateway override.
    use_exchange_as_venue : bool, default True
        If the `exchange` field will be used as the venue for instrument IDs.

    Returns
    -------
    bomber_pyo3.DatabentoHistoricalClient

    """
    return bomber_pyo3.DatabentoHistoricalClient(
        key=key or get_env_key("DATABENTO_API_KEY"),
        publishers_filepath=str(PUBLISHERS_FILEPATH),
        use_exchange_as_venue=use_exchange_as_venue,
    )


@lru_cache(1)
def get_cached_databento_instrument_provider(
    http_client: bomber_pyo3.DatabentoHistoricalClient,
    clock: LiveClock,
    live_api_key: str | None = None,
    live_gateway: str | None = None,
    loader: DatabentoDataLoader | None = None,
    config: InstrumentProviderConfig | None = None,
    use_exchange_as_venue=True,
) -> DatabentoInstrumentProvider:
    """
    Cache and return a Databento instrument provider.

    If a cached provider already exists, then that provider will be returned.

    Parameters
    ----------
    http_client : bomber_pyo3.DatabentoHistoricalClient
        The client for the instrument provider.
    clock : LiveClock
        The clock for the instrument provider.
    live_api_key : str, optional
        The specific API secret key for Databento live clients.
        If not provided then will use the historical HTTP client API key.
    live_gateway : str, optional
        The live gateway override for Databento live clients.
    loader : DatabentoDataLoader, optional
        The loader for the provider.
    config : InstrumentProviderConfig
        The configuration for the instrument provider.
    use_exchange_as_venue : bool, default True
        If the `exchange` field will be used as the venue for instrument IDs.

    Returns
    -------
    DatabentoInstrumentProvider

    """
    return DatabentoInstrumentProvider(
        http_client=http_client,
        clock=clock,
        live_api_key=live_api_key,
        live_gateway=live_gateway,
        loader=loader,
        config=config,
        use_exchange_as_venue=use_exchange_as_venue,
    )


class DatabentoLiveDataClientFactory(LiveDataClientFactory):
    """
    Provides a Binance live data client factory.
    """

    @staticmethod
    def create(  # type: ignore
        loop: asyncio.AbstractEventLoop,
        name: str,
        config: DatabentoDataClientConfig,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> DatabentoDataClient:
        """
        Create a new Databento data client.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The event loop for the client.
        name : str
            The custom client name.
        config : DatabentoDataClientConfig
            The client configuration.
        msgbus : MessageBus
            The message bus for the client.
        cache : Cache
            The cache for the client.
        clock : LiveClock
            The clock for the client.

        Returns
        -------
        DatabentoDataClient

        """
        # Get HTTP client singleton
        http_client = get_cached_databento_http_client(
            key=config.api_key,
            gateway=config.http_gateway,
            use_exchange_as_venue=config.use_exchange_as_venue,
        )

        loader = DatabentoDataLoader(config.venue_dataset_map)
        provider = get_cached_databento_instrument_provider(
            http_client=http_client,
            clock=clock,
            live_api_key=config.api_key,
            live_gateway=config.live_gateway,
            loader=loader,
            config=config.instrument_provider,
            use_exchange_as_venue=config.use_exchange_as_venue,
        )

        return DatabentoDataClient(
            loop=loop,
            http_client=http_client,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=provider,
            loader=loader,
            config=config,
            name=name,
        )
