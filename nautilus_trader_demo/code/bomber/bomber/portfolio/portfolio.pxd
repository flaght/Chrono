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

from libc.stdint cimport uint64_t

from bomber.accounting.accounts.base cimport Account
from bomber.accounting.manager cimport AccountsManager
from bomber.cache.cache cimport Cache
from bomber.common.component cimport Clock
from bomber.common.component cimport Logger
from bomber.common.component cimport MessageBus
from bomber.core.rust.model cimport OrderSide
from bomber.core.rust.model cimport PriceType
from bomber.model.data cimport Bar
from bomber.model.data cimport QuoteTick
from bomber.model.events.account cimport AccountState
from bomber.model.events.order cimport OrderEvent
from bomber.model.events.position cimport PositionEvent
from bomber.model.identifiers cimport AccountId
from bomber.model.identifiers cimport InstrumentId
from bomber.model.identifiers cimport PositionId
from bomber.model.identifiers cimport Venue
from bomber.model.instruments.base cimport Instrument
from bomber.model.objects cimport Currency
from bomber.model.objects cimport Money
from bomber.model.objects cimport Price
from bomber.model.position cimport Position
from bomber.portfolio.base cimport PortfolioFacade


cdef class Portfolio(PortfolioFacade):
    cdef Clock _clock
    cdef Logger _log
    cdef MessageBus _msgbus
    cdef Cache _cache
    cdef AccountsManager _accounts
    cdef object _config
    cdef bint _debug
    cdef bint _use_mark_prices
    cdef bint _use_mark_xrates
    cdef bint _convert_to_account_base_currency
    cdef uint64_t _min_account_state_logging_interval_ns
    cdef str _log_price
    cdef str _log_xrate

    cdef dict[InstrumentId, dict[AccountId, Money]] _unrealized_pnls
    cdef dict[InstrumentId, dict[AccountId, Money]] _realized_pnls
    cdef dict[PositionId, Money] _snapshot_sum_per_position
    cdef dict[PositionId, Money] _snapshot_last_per_position
    cdef dict[PositionId, int] _snapshot_processed_counts
    cdef dict[PositionId, AccountId] _snapshot_account_ids
    cdef dict[InstrumentId, dict[AccountId, Decimal]] _net_positions
    cdef dict[PositionId, object] _bet_positions
    cdef object _index_bet_positions
    cdef set[InstrumentId] _pending_calcs
    cdef dict[InstrumentId, Price] _bar_close_prices
    cdef dict[AccountId, uint64_t] _last_account_state_log_ts
    cdef dict[Venue, set] _venues_missing_price

    # -- COMMANDS -------------------------------------------------------------------------------------

    cpdef void set_use_mark_prices(self, bint value)
    cpdef void set_use_mark_xrates(self, bint value)
    cpdef void initialize_orders(self)
    cpdef void initialize_positions(self)
    cpdef void update_quote_tick(self, QuoteTick tick)
    cpdef void update_mark_price(self, object mark_price)
    cpdef void update_bar(self, Bar bar)
    cpdef void update_account(self, AccountState event)
    cpdef void update_order(self, OrderEvent event)
    cpdef void update_position(self, PositionEvent event)
    cpdef void on_order_event(self, OrderEvent event)
    cpdef void on_position_event(self, PositionEvent event)

    # -- INTERNAL -------------------------------------------------------------------------------------

    cdef void _update_account(self, AccountState event)
    cdef Account _get_account(self, Venue venue, AccountId account_id, str caller_name, str message=*)
    cdef Venue _accumulate_mark_values(self, Venue venue, AccountId account_id, dict values, set unpriced)
    cdef void _update_missing_price_state(self, Venue venue, set unpriced)
    cdef void _update_mark_xrate(self, Instrument instrument, double xrate, InstrumentId instrument_id)
    cdef void _update_instrument_id(self, InstrumentId instrument_id)
    cdef void _update_net_position(self, InstrumentId instrument_id, list positions_open)
    cdef object _net_position(self, InstrumentId instrument_id, AccountId account_id=*)
    cdef void _ensure_snapshot_pnls_cached_for(self, InstrumentId instrument_id)
    cdef Price _get_price(self, Position position)
    cdef object _get_xrate_to_account_base(self, Instrument instrument, Account account, InstrumentId instrument_id)
    cdef dict _group_by_account_id(self, list items)
    cdef Money _add_pnl_to_total(self, Money total_pnl, Money pnl, str pnl_type, Venue venue=*, Currency target_currency=*)
    cdef Money _aggregate_pnl_from_cache(self, InstrumentId instrument_id, bint is_realized, Currency target_currency=*)
    cdef Money _aggregate_pnl_by_calculation(self, InstrumentId instrument_id, Price price, bint is_realized, Currency target_currency=*)
    cdef Money _convert_money(self, Money money, Currency target_currency, Venue venue=*, PriceType price_type=*)
    cdef Money _convert_money_if_needed(self, Money money, Currency target_currency, Venue venue=*, PriceType price_type=*)
    cdef object _get_bet_position(self, Position position, Instrument instrument)
    cdef Money _get_zero_or_none_for_instrument(self, InstrumentId instrument_id, Currency target_currency=*)
    cdef tuple _validate_event_account_and_instrument(self, object event, str caller_name)
    cdef Money _calculate_realized_pnl(self, InstrumentId instrument_id, AccountId account_id)
    cdef tuple _validate_account_and_instrument(self, InstrumentId instrument_id, AccountId account_id, str caller_name, bint is_error)
    cdef Currency _determine_pnl_currency(self, Account account, Instrument instrument)
    cdef dict _aggregate_pnls_by_instrument(self, list positions, bint is_realized, AccountId account_id, Currency target_currency)
    cdef tuple _process_snapshot_pnl_contributions(self, InstrumentId instrument_id, AccountId account_id, list positions, Currency currency, Account account)
    cdef object _calculate_snapshot_contribution(self, PositionId position_id, set active_position_ids, list positions, Money sum_pnl, set processed_ids)
    cdef object _process_active_position_realized_pnl(self, list positions, InstrumentId instrument_id, Instrument instrument, Account account, Currency currency, set processed_ids)
    cdef Money _calculate_unrealized_pnl(self, InstrumentId instrument_id, Price price=*, AccountId account_id=*)
    cdef object _calculate_total_unrealized_pnl(self, list positions_open, InstrumentId instrument_id, Instrument instrument, Account account, Currency currency, Price price)
    cdef object _calculate_position_unrealized_pnl(self, Position position, Instrument instrument, Account account, Currency currency, InstrumentId instrument_id, Price price)
    cdef double _handle_betting_instrument_exposure(self, list positions, Instrument instrument_obj, Currency target_currency, Currency settlement_currency)
    cdef tuple _calculate_non_betting_exposure(self, list positions, InstrumentId instrument_id, Instrument instrument_obj, Price price, Currency target_currency)
    cdef tuple _calculate_position_exposure_value(self, Position position, Instrument instrument_obj, InstrumentId instrument_id, Price price, Currency target_currency, bint is_currency_pair, list positions)
    cdef object _calculate_currency_pair_exposure(self, Position position, Instrument instrument_obj, InstrumentId instrument_id, Price price, Currency target_currency, list positions, Price price_param)
