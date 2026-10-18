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

from cpython.datetime cimport datetime

from bomber.cache.cache cimport Cache
from bomber.common.component cimport Component
from bomber.core.data cimport Data
from bomber.core.uuid cimport UUID4
from bomber.data.messages cimport RequestBars
from bomber.data.messages cimport RequestData
from bomber.data.messages cimport RequestForwardPrices
from bomber.data.messages cimport RequestFundingRates
from bomber.data.messages cimport RequestInstrument
from bomber.data.messages cimport RequestInstruments
from bomber.data.messages cimport RequestOrderBookDeltas
from bomber.data.messages cimport RequestOrderBookSnapshot
from bomber.data.messages cimport RequestQuoteTicks
from bomber.data.messages cimport RequestTradeTicks
from bomber.data.messages cimport SubscribeBars
from bomber.data.messages cimport SubscribeData
from bomber.data.messages cimport SubscribeFundingRates
from bomber.data.messages cimport SubscribeIndexPrices
from bomber.data.messages cimport SubscribeInstrument
from bomber.data.messages cimport SubscribeInstrumentClose
from bomber.data.messages cimport SubscribeInstruments
from bomber.data.messages cimport SubscribeInstrumentStatus
from bomber.data.messages cimport SubscribeMarkPrices
from bomber.data.messages cimport SubscribeOptionGreeks
from bomber.data.messages cimport SubscribeOrderBook
from bomber.data.messages cimport SubscribeQuoteTicks
from bomber.data.messages cimport SubscribeTradeTicks
from bomber.data.messages cimport UnsubscribeBars
from bomber.data.messages cimport UnsubscribeData
from bomber.data.messages cimport UnsubscribeFundingRates
from bomber.data.messages cimport UnsubscribeIndexPrices
from bomber.data.messages cimport UnsubscribeInstrument
from bomber.data.messages cimport UnsubscribeInstrumentClose
from bomber.data.messages cimport UnsubscribeInstruments
from bomber.data.messages cimport UnsubscribeInstrumentStatus
from bomber.data.messages cimport UnsubscribeMarkPrices
from bomber.data.messages cimport UnsubscribeOptionGreeks
from bomber.data.messages cimport UnsubscribeOrderBook
from bomber.data.messages cimport UnsubscribeQuoteTicks
from bomber.data.messages cimport UnsubscribeTradeTicks
from bomber.model.data cimport Bar
from bomber.model.data cimport BarType
from bomber.model.data cimport DataType
from bomber.model.identifiers cimport InstrumentId
from bomber.model.identifiers cimport Venue
from bomber.model.instruments.base cimport Instrument


cdef class DataClient(Component):
    cdef readonly Cache _cache
    cdef set _subscriptions_generic

    cdef readonly Venue venue
    """The clients venue ID (if applicable).\n\n:returns: `Venue` or ``None``"""
    cdef readonly bint is_connected
    """If the client is connected.\n\n:returns: `bool`"""

    cpdef void _set_connected(self, bint value=*)

# -- SUBSCRIPTIONS --------------------------------------------------------------------------------

    cpdef list subscribed_custom_data(self)

    cpdef void subscribe(self, SubscribeData command)
    cpdef void unsubscribe(self, UnsubscribeData command)

    cpdef void _add_subscription(self, DataType data_type)
    cpdef void _remove_subscription(self, DataType data_type)

# -- REQUEST HANDLERS -----------------------------------------------------------------------------

    cpdef void request(self, RequestData request)

# -- DATA HANDLERS --------------------------------------------------------------------------------

    cpdef void _handle_data(self, Data data)
    cpdef void _handle_data_response(self, DataType data_type, data, UUID4 correlation_id, datetime start, datetime end, dict[str, object] params)


cdef class MarketDataClient(DataClient):
    cdef set[InstrumentId] _subscriptions_order_book_delta
    cdef set[InstrumentId] _subscriptions_order_book_depth
    cdef set[InstrumentId] _subscriptions_quote_tick
    cdef set[InstrumentId] _subscriptions_trade_tick
    cdef set[InstrumentId] _subscriptions_mark_price
    cdef set[InstrumentId] _subscriptions_index_price
    cdef set[InstrumentId] _subscriptions_funding_rate
    cdef set[InstrumentId] _subscriptions_instrument_status
    cdef set[InstrumentId] _subscriptions_instrument_close
    cdef set[InstrumentId] _subscriptions_instrument
    cdef set[InstrumentId] _subscriptions_option_greeks
    cdef set[BarType] _subscriptions_bar

    cdef object _update_instruments_task

# -- SUBSCRIPTIONS --------------------------------------------------------------------------------

    cpdef list subscribed_instruments(self)
    cpdef list subscribed_order_book_deltas(self)
    cpdef list subscribed_order_book_depth(self)
    cpdef list subscribed_quote_ticks(self)
    cpdef list subscribed_trade_ticks(self)
    cpdef list subscribed_mark_prices(self)
    cpdef list subscribed_index_prices(self)
    cpdef list subscribed_funding_rates(self)
    cpdef list subscribed_bars(self)
    cpdef list subscribed_instrument_status(self)
    cpdef list subscribed_instrument_close(self)
    cpdef list subscribed_option_greeks(self)

    cpdef bint is_subscribed_order_book_deltas(self, InstrumentId instrument_id)
    cpdef bint is_subscribed_quote_ticks(self, InstrumentId instrument_id)
    cpdef bint is_subscribed_trade_ticks(self, InstrumentId instrument_id)

    cpdef void subscribe_instruments(self, SubscribeInstruments command)
    cpdef void subscribe_instrument(self, SubscribeInstrument command)
    cpdef void subscribe_order_book_deltas(self, SubscribeOrderBook command)
    cpdef void subscribe_order_book_depth(self, SubscribeOrderBook command)
    cpdef void subscribe_quote_ticks(self, SubscribeQuoteTicks command)
    cpdef void subscribe_trade_ticks(self, SubscribeTradeTicks command)
    cpdef void subscribe_mark_prices(self, SubscribeMarkPrices command)
    cpdef void subscribe_index_prices(self, SubscribeIndexPrices command)
    cpdef void subscribe_funding_rates(self, SubscribeFundingRates command)
    cpdef void subscribe_bars(self, SubscribeBars command)
    cpdef void subscribe_instrument_status(self, SubscribeInstrumentStatus command)
    cpdef void subscribe_instrument_close(self, SubscribeInstrumentClose command)
    cpdef void subscribe_option_greeks(self, SubscribeOptionGreeks command)
    cpdef void unsubscribe_instruments(self, UnsubscribeInstruments command)
    cpdef void unsubscribe_instrument(self, UnsubscribeInstrument command)
    cpdef void unsubscribe_order_book_deltas(self, UnsubscribeOrderBook command)
    cpdef void unsubscribe_order_book_depth(self, UnsubscribeOrderBook command)
    cpdef void unsubscribe_quote_ticks(self, UnsubscribeQuoteTicks command)
    cpdef void unsubscribe_trade_ticks(self, UnsubscribeTradeTicks command)
    cpdef void unsubscribe_mark_prices(self, UnsubscribeMarkPrices command)
    cpdef void unsubscribe_index_prices(self, UnsubscribeIndexPrices command)
    cpdef void unsubscribe_funding_rates(self, UnsubscribeFundingRates command)
    cpdef void unsubscribe_bars(self, UnsubscribeBars command)
    cpdef void unsubscribe_instrument_status(self, UnsubscribeInstrumentStatus command)
    cpdef void unsubscribe_instrument_close(self, UnsubscribeInstrumentClose command)
    cpdef void unsubscribe_option_greeks(self, UnsubscribeOptionGreeks command)

    cpdef void _add_subscription_instrument(self, InstrumentId instrument_id)
    cpdef void _add_subscription_order_book_deltas(self, InstrumentId instrument_id)
    cpdef void _add_subscription_order_book_depth(self, InstrumentId instrument_id)
    cpdef void _add_subscription_quote_ticks(self, InstrumentId instrument_id)
    cpdef void _add_subscription_trade_ticks(self, InstrumentId instrument_id)
    cpdef void _add_subscription_mark_prices(self, InstrumentId instrument_id)
    cpdef void _add_subscription_index_prices(self, InstrumentId instrument_id)
    cpdef void _add_subscription_funding_rates(self, InstrumentId instrument_id)
    cpdef void _add_subscription_bars(self, BarType bar_type)
    cpdef void _add_subscription_instrument_status(self, InstrumentId instrument_id)
    cpdef void _add_subscription_instrument_close(self, InstrumentId instrument_id)
    cpdef void _add_subscription_option_greeks(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_instrument(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_order_book_deltas(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_order_book_depth(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_quote_ticks(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_trade_ticks(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_mark_prices(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_index_prices(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_funding_rates(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_bars(self, BarType bar_type)
    cpdef void _remove_subscription_instrument_status(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_instrument_close(self, InstrumentId instrument_id)
    cpdef void _remove_subscription_option_greeks(self, InstrumentId instrument_id)

# -- REQUEST HANDLERS -----------------------------------------------------------------------------

    cpdef void request_instrument(self, RequestInstrument request)
    cpdef void request_instruments(self, RequestInstruments request)
    cpdef void request_order_book_deltas(self, RequestOrderBookDeltas request)
    cpdef void request_order_book_snapshot(self, RequestOrderBookSnapshot request)
    cpdef void request_quote_ticks(self, RequestQuoteTicks request)
    cpdef void request_trade_ticks(self, RequestTradeTicks request)
    cpdef void request_funding_rates(self, RequestFundingRates request)
    cpdef void request_bars(self, RequestBars request)
    cpdef void request_forward_prices(self, RequestForwardPrices request)

# -- DATA HANDLERS --------------------------------------------------------------------------------

    cpdef void _handle_instrument(self, Instrument instrument, UUID4 correlation_id, datetime start, datetime end, dict[str, object] params)
    cpdef void _handle_instruments(self, Venue venue, list instruments, UUID4 correlation_id, datetime start, datetime end, dict[str, object] params)
    cpdef void _handle_quote_ticks(self, InstrumentId instrument_id, list ticks, UUID4 correlation_id, datetime start, datetime end, dict[str, object] params)
    cpdef void _handle_trade_ticks(self, InstrumentId instrument_id, list ticks, UUID4 correlation_id, datetime start, datetime end, dict[str, object] params)
    cpdef void _handle_funding_rates(self, InstrumentId instrument_id, list funding_rates, UUID4 correlation_id, datetime start, datetime end, dict[str, object] params)
    cpdef void _handle_bars(self, BarType bar_type, list bars, UUID4 correlation_id, datetime start, datetime end, dict[str, object] params)
    cpdef void _handle_order_book_depths(self, InstrumentId instrument_id, list depths, UUID4 correlation_id, datetime start, datetime end, dict[str, object] params)
    cpdef void _handle_order_book_deltas(self, InstrumentId instrument_id, list deltas, UUID4 correlation_id, datetime start, datetime end, dict[str, object] params)
    cpdef void _handle_forward_prices(self, list forward_prices, UUID4 correlation_id, dict[str, object] params)
