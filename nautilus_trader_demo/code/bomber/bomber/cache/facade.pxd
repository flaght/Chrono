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
from libc.stdint cimport uint64_t

from bomber.accounting.accounts.base cimport Account
from bomber.common.actor cimport Actor
from bomber.common.component cimport Logger
from bomber.core.rust.model cimport OrderSide
from bomber.core.rust.model cimport PositionSide
from bomber.core.rust.model cimport PriceType
from bomber.execution.messages cimport SubmitOrder
from bomber.execution.messages cimport SubmitOrderList
from bomber.model.book cimport OrderBook
from bomber.model.data cimport Bar
from bomber.model.data cimport BarType
from bomber.model.data cimport QuoteTick
from bomber.model.data cimport TradeTick
from bomber.model.identifiers cimport AccountId
from bomber.model.identifiers cimport ClientId
from bomber.model.identifiers cimport ClientOrderId
from bomber.model.identifiers cimport ComponentId
from bomber.model.identifiers cimport ExecAlgorithmId
from bomber.model.identifiers cimport InstrumentId
from bomber.model.identifiers cimport OrderListId
from bomber.model.identifiers cimport PositionId
from bomber.model.identifiers cimport StrategyId
from bomber.model.identifiers cimport Venue
from bomber.model.identifiers cimport VenueOrderId
from bomber.model.instruments.base cimport Instrument
from bomber.model.instruments.synthetic cimport SyntheticInstrument
from bomber.model.objects cimport Currency
from bomber.model.objects cimport Money
from bomber.model.objects cimport Price
from bomber.model.objects cimport Quantity
from bomber.model.orders.base cimport Order
from bomber.model.orders.list cimport OrderList
from bomber.model.position cimport Position
from bomber.trading.strategy cimport Strategy


cdef class CacheDatabaseFacade:
    cdef Logger _log

    cpdef void close(self)
    cpdef void flush(self)
    cpdef list[str] keys(self, str pattern=*)
    cpdef dict load_all(self)
    cpdef dict load(self)
    cpdef dict load_currencies(self)
    cpdef dict load_instruments(self)
    cpdef dict load_synthetics(self)
    cpdef dict load_accounts(self)
    cpdef dict load_orders(self)
    cpdef dict load_positions(self)
    cpdef dict load_index_order_position(self)
    cpdef dict load_index_order_client(self)
    cpdef Currency load_currency(self, str code)
    cpdef Instrument load_instrument(self, InstrumentId instrument_id)
    cpdef SyntheticInstrument load_synthetic(self, InstrumentId instrument_id)
    cpdef Account load_account(self, AccountId account_id)
    cpdef Order load_order(self, ClientOrderId order_id)
    cpdef Position load_position(self, PositionId position_id)
    cpdef dict load_actor(self, ComponentId component_id)
    cpdef dict load_strategy(self, StrategyId strategy_id)

    cpdef void add(self, str key, bytes value)
    cpdef void add_currency(self, Currency currency)
    cpdef void add_instrument(self, Instrument instrument)
    cpdef void add_synthetic(self, SyntheticInstrument instrument)
    cpdef void add_account(self, Account account)
    cpdef void add_order(self, Order order, PositionId position_id=*, ClientId client_id=*)
    cpdef void add_position(self, Position position)

    cpdef void index_venue_order_id(self, ClientOrderId client_order_id, VenueOrderId venue_order_id)
    cpdef void index_order_position(self, ClientOrderId client_order_id, PositionId position_id)

    cpdef void update_account(self, Account account)
    cpdef void update_order(self, Order order)
    cpdef void update_position(self, Position position)
    cpdef void update_actor(self, Actor actor)
    cpdef void update_strategy(self, Strategy strategy)

    cpdef void snapshot_order_state(self, Order order)
    cpdef void snapshot_position_state(self, Position position, uint64_t ts_snapshot, Money unrealized_pnl=*)

    cpdef void delete_order(self, ClientOrderId client_order_id)
    cpdef void delete_position(self, PositionId position_id)
    cpdef void delete_account_event(self, AccountId account_id, str event_id)
    cpdef void delete_actor(self, ComponentId component_id)
    cpdef void delete_strategy(self, StrategyId strategy_id)

    cpdef void heartbeat(self, datetime timestamp)
