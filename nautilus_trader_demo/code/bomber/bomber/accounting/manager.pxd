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
from bomber.accounting.accounts.cash cimport CashAccount
from bomber.accounting.accounts.margin cimport MarginAccount
from bomber.cache.base cimport CacheFacade
from bomber.common.component cimport Clock
from bomber.common.component cimport Logger
from bomber.core.rust.model cimport OrderSide
from bomber.model.events.account cimport AccountState
from bomber.model.events.order cimport OrderFilled
from bomber.model.instruments.base cimport Instrument
from bomber.model.objects cimport Money


cdef class AccountsManager:
    cdef Clock _clock
    cdef Logger _log
    cdef CacheFacade _cache

    cpdef AccountState generate_account_state(self, Account account, uint64_t ts_event)
    cpdef void update_balances(self, Account account, Instrument instrument, OrderFilled fill)
    cpdef bint update_orders(self, Account account, Instrument instrument, list orders_open, uint64_t ts_event)
    cpdef bint update_positions(self, MarginAccount account, Instrument instrument, list positions_open, uint64_t ts_event)
    cdef bint _update_balance_locked(self, CashAccount account, Instrument instrument, list orders_open, uint64_t ts_event)
    cdef bint _update_margin_init(self, MarginAccount account, Instrument instrument, list orders_open, uint64_t ts_event)
    cdef bint _update_balance_single_currency(self, Account account, OrderFilled fill, Money pnl)
    cdef bint _update_balance_multi_currency(self, Account account, OrderFilled fill, list pnls)
    cdef object _calculate_xrate_to_base(self, Account account, Instrument instrument, OrderSide side)
