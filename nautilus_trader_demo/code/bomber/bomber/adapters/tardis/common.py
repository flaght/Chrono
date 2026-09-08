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

import datetime as dt

import msgspec

from bomber.core import bomber_pyo3
from bomber.core.correctness import PyCondition
from bomber.model.data import BarType
from bomber.model.data import FundingRateUpdate
from bomber.model.data import OrderBookDelta
from bomber.model.data import OrderBookDepth10
from bomber.model.data import QuoteTick
from bomber.model.data import TradeTick
from bomber.model.identifiers import InstrumentId
from bomber.model.instruments import CryptoFuture
from bomber.model.instruments import CryptoOption
from bomber.model.instruments import CryptoPerpetual
from bomber.model.instruments import CurrencyPair
from bomber.model.instruments import Instrument


def create_instrument_info(instrument: Instrument) -> bomber_pyo3.TardisInstrumentMiniInfo:
    return bomber_pyo3.TardisInstrumentMiniInfo(
        instrument_id=bomber_pyo3.InstrumentId.from_str(instrument.id.value),
        raw_symbol=instrument.raw_symbol.value,
        exchange=infer_tardis_exchange_str(instrument),
        price_precision=instrument.price_precision,
        size_precision=instrument.size_precision,
    )


def infer_tardis_exchange_str(instrument: Instrument) -> str:  # noqa: C901 (too complex)
    venue = instrument.venue.value

    match venue:
        case "BINANCE":
            if isinstance(instrument, CurrencyPair):
                return "binance"
            elif isinstance(instrument, CryptoOption):
                return "binance-options"
            else:
                return "binance-futures"
        case "BINANCE_US":
            return "binance-us"
        case "BINANCE_DELIVERY":
            return "binance-delivery"
        case "BITFINEX":
            if isinstance(instrument, CurrencyPair):
                return "bitfinex"
            else:
                return "bitfinex-derivatives"
        case "BYBIT":
            if isinstance(instrument, CurrencyPair):
                return "bybit-spot"
            elif isinstance(instrument, CryptoOption):
                return "bybit-options"
            else:
                return "bybit"
        case "CRYPTO_COM":
            if isinstance(instrument, CurrencyPair):
                return "crypto-com"
        case "GATE_IO":
            if isinstance(instrument, CurrencyPair):
                return "gate-io"
            else:
                return "gate-io-futures"
        case "HUOBI":
            if isinstance(instrument, CurrencyPair):
                return "huobi"
            elif isinstance(instrument, CryptoPerpetual):
                return "huobi-dm-linear-swap"
            elif isinstance(instrument, CryptoFuture):
                return "huobi-dm"
            elif isinstance(instrument, CryptoOption):
                return "huobi-dm-options"
        case "HUOBI_DELIVERY":
            return "huobi-dm-swap"
        case "OKEX":
            if isinstance(instrument, CurrencyPair):
                return "okex"
            elif isinstance(instrument, CryptoPerpetual):
                return "okex-swap"
            elif isinstance(instrument, CryptoFuture):
                return "okex-futures"
            elif isinstance(instrument, CryptoOption):
                return "okex-options"
        case "COINBASE_INTX":
            return "coinbase-international"
        case "BITGET":
            if isinstance(instrument, CurrencyPair):
                return "bitget"
            else:
                return "bitget-futures"

    return venue.lower().replace("_", "-")


def get_ws_client_key(instrument_id: InstrumentId, tardis_data_type: str) -> str:
    return f"{instrument_id}-{tardis_data_type}"


def convert_nautilus_data_type_to_tardis_data_type(data_type: type) -> str:
    if data_type is OrderBookDelta:
        return "book_change"
    elif data_type is OrderBookDepth10:
        return "book_snapshot"
    elif data_type is QuoteTick:
        return "quote"
    elif data_type is TradeTick:
        return "trade"
    elif data_type is FundingRateUpdate:
        return "derivative_ticker"
    else:
        raise ValueError(f"Invalid `data_type` to convert, was {data_type}")


def convert_nautilus_bar_type_to_tardis_data_type(bar_type: BarType) -> str:
    bar_type_pyo3 = bomber_pyo3.BarType.from_str(str(bar_type))
    return bomber_pyo3.bar_spec_to_tardis_trade_bar_string(bar_type_pyo3.spec)


def create_replay_normalized_request_options(
    exchange: str,
    symbols: list[str],
    from_date: dt.date,
    to_date: dt.date,
    data_types: list[str],
) -> bomber_pyo3.ReplayNormalizedRequestOptions:
    PyCondition.not_empty(symbols, "symbols")
    PyCondition.not_empty(data_types, "data_types")

    options = {
        "exchange": exchange,
        "symbols": symbols,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "data_types": data_types,
        "with_disconnect_messages": True,
    }

    json_options = msgspec.json.encode(options)
    return bomber_pyo3.ReplayNormalizedRequestOptions.from_json(json_options)


def create_stream_normalized_request_options(
    exchange: str,
    symbols: list[str],
    data_types: list[str],
) -> bomber_pyo3.StreamNormalizedRequestOptions:
    PyCondition.not_empty(symbols, "symbols")
    PyCondition.not_empty(data_types, "data_types")

    options = {
        "exchange": exchange,
        "symbols": symbols,
        "data_types": data_types,
        "with_disconnect_messages": True,
    }

    json_options = msgspec.json.encode(options)
    return bomber_pyo3.StreamNormalizedRequestOptions.from_json(json_options)
