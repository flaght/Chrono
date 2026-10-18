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
from libc.stdint cimport uint8_t
from libc.stdint cimport uint64_t

from bomber.cache.base cimport CacheFacade
from bomber.common.actor cimport Actor
from bomber.common.component cimport Clock
from bomber.common.component cimport MessageBus
from bomber.core.message cimport Command
from bomber.core.rust.model cimport ContingencyType
from bomber.core.rust.model cimport TimeInForce
from bomber.core.rust.model cimport TriggerType
from bomber.execution.messages cimport CancelOrder
from bomber.execution.messages cimport ModifyOrder
from bomber.execution.messages cimport SubmitOrder
from bomber.execution.messages cimport SubmitOrderList
from bomber.execution.messages cimport TradingCommand
from bomber.model.events.order cimport Event
from bomber.model.events.order cimport OrderAccepted
from bomber.model.events.order cimport OrderCanceled
from bomber.model.events.order cimport OrderCancelRejected
from bomber.model.events.order cimport OrderDenied
from bomber.model.events.order cimport OrderEmulated
from bomber.model.events.order cimport OrderEvent
from bomber.model.events.order cimport OrderExpired
from bomber.model.events.order cimport OrderFilled
from bomber.model.events.order cimport OrderInitialized
from bomber.model.events.order cimport OrderModifyRejected
from bomber.model.events.order cimport OrderPendingCancel
from bomber.model.events.order cimport OrderPendingUpdate
from bomber.model.events.order cimport OrderRejected
from bomber.model.events.order cimport OrderReleased
from bomber.model.events.order cimport OrderSubmitted
from bomber.model.events.order cimport OrderTriggered
from bomber.model.events.order cimport OrderUpdated
from bomber.model.events.position cimport PositionChanged
from bomber.model.events.position cimport PositionClosed
from bomber.model.events.position cimport PositionEvent
from bomber.model.events.position cimport PositionOpened
from bomber.model.identifiers cimport ClientId
from bomber.model.identifiers cimport ClientOrderId
from bomber.model.identifiers cimport PositionId
from bomber.model.identifiers cimport StrategyId
from bomber.model.identifiers cimport TraderId
from bomber.model.objects cimport Price
from bomber.model.objects cimport Quantity
from bomber.model.orders.base cimport Order
from bomber.model.orders.limit cimport LimitOrder
from bomber.model.orders.list cimport OrderList
from bomber.model.orders.market cimport MarketOrder
from bomber.model.orders.market_to_limit cimport MarketToLimitOrder
from bomber.portfolio.base cimport PortfolioFacade


cdef class ExecAlgorithm(Actor):
    cdef dict[ClientOrderId, int] _exec_spawn_ids
    cdef set[StrategyId] _subscribed_strategies
    cdef dict[ClientOrderId, Quantity] _pending_spawn_reductions

# -- REGISTRATION ---------------------------------------------------------------------------------

    cpdef void register(
        self,
        TraderId trader_id,
        PortfolioFacade portfolio,
        MessageBus msgbus,
        CacheFacade cache,
        Clock clock,
    )

# -- INTERNAL -------------------------------------------------------------------------------------

    cdef ClientOrderId _spawn_client_order_id(self, Order primary)
    cdef void _reduce_primary_order(self, Order primary, Quantity spawn_qty)
    cdef void _restore_primary_order_quantity(self, Order order)

# -- COMMANDS -------------------------------------------------------------------------------------

    cpdef void execute(self, TradingCommand command)
    cdef void _handle_submit_order(self, SubmitOrder command)
    cdef void _handle_submit_order_list(self, SubmitOrderList command)
    cdef void _handle_cancel_order(self, CancelOrder command)

# -- EVENT HANDLERS -------------------------------------------------------------------------------

    cdef void _handle_event(self, Event event)
    cpdef void on_order(self, Order order)
    cpdef void on_order_list(self, OrderList order_list)
    cpdef void on_order_event(self, OrderEvent event)
    cpdef void on_order_initialized(self, OrderInitialized event)
    cpdef void on_order_denied(self, OrderDenied event)
    cpdef void on_order_emulated(self, OrderEmulated event)
    cpdef void on_order_released(self, OrderReleased event)
    cpdef void on_order_submitted(self, OrderSubmitted event)
    cpdef void on_order_rejected(self, OrderRejected event)
    cpdef void on_order_accepted(self, OrderAccepted event)
    cpdef void on_order_canceled(self, OrderCanceled event)
    cpdef void on_order_expired(self, OrderExpired event)
    cpdef void on_order_triggered(self, OrderTriggered event)
    cpdef void on_order_pending_update(self, OrderPendingUpdate event)
    cpdef void on_order_pending_cancel(self, OrderPendingCancel event)
    cpdef void on_order_modify_rejected(self, OrderModifyRejected event)
    cpdef void on_order_cancel_rejected(self, OrderCancelRejected event)
    cpdef void on_order_updated(self, OrderUpdated event)
    cpdef void on_order_filled(self, OrderFilled event)
    cpdef void on_position_event(self, PositionEvent event)
    cpdef void on_position_opened(self, PositionOpened event)
    cpdef void on_position_changed(self, PositionChanged event)
    cpdef void on_position_closed(self, PositionClosed event)

# -- TRADING COMMANDS -----------------------------------------------------------------------------

    cpdef MarketOrder spawn_market(
        self,
        Order primary,
        Quantity quantity,
        TimeInForce time_in_force=*,
        bint reduce_only=*,
        list[str] tags=*,
        bint reduce_primary=*,
    )

    cpdef LimitOrder spawn_limit(
        self,
        Order primary,
        Quantity quantity,
        Price price,
        TimeInForce time_in_force=*,
        datetime expire_time=*,
        bint post_only=*,
        bint reduce_only=*,
        Quantity display_qty=*,
        TriggerType emulation_trigger=*,
        list[str] tags=*,
        bint reduce_primary=*,
    )

    cpdef MarketToLimitOrder spawn_market_to_limit(
        self,
        Order primary,
        Quantity quantity,
        TimeInForce time_in_force=*,
        datetime expire_time=*,
        bint reduce_only=*,
        Quantity display_qty=*,
        TriggerType emulation_trigger=*,
        list[str] tags=*,
        bint reduce_primary=*,
    )

    cpdef void submit_order(self, Order order)
    cpdef void modify_order(
        self,
        Order order,
        Quantity quantity=*,
        Price price=*,
        Price trigger_price=*,
        ClientId client_id=*,
    )
    cpdef void modify_order_in_place(
        self,
        Order order,
        Quantity quantity=*,
        Price price=*,
        Price trigger_price=*,
    )
    cpdef void cancel_order(self, Order order, ClientId client_id=*)

# -- EVENTS ---------------------------------------------------------------------------------------

    cdef OrderPendingUpdate _generate_order_pending_update(self, Order order)
    cdef OrderPendingCancel _generate_order_pending_cancel(self, Order order)
    cdef OrderCanceled _generate_order_canceled(self, Order order)

# -- EGRESS ---------------------------------------------------------------------------------------

    cdef void _send_emulator_command(self, TradingCommand command)
    cdef void _send_risk_command(self, TradingCommand command)
    cdef void _send_exec_command(self, Command command)
