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

from typing import Any
from typing import Callable

from bomber.common.messages cimport ComponentStateChanged
from bomber.common.messages cimport ShutdownSystem
from bomber.common.messages cimport TradingStateChanged
from bomber.core.correctness cimport Condition
from bomber.execution.messages cimport BatchCancelOrders
from bomber.execution.messages cimport CancelAllOrders
from bomber.execution.messages cimport CancelOrder
from bomber.execution.messages cimport GenerateExecutionMassStatus
from bomber.execution.messages cimport GenerateFillReports
from bomber.execution.messages cimport GenerateOrderStatusReport
from bomber.execution.messages cimport GenerateOrderStatusReports
from bomber.execution.messages cimport GeneratePositionStatusReports
from bomber.execution.messages cimport ModifyOrder
from bomber.execution.messages cimport QueryAccount
from bomber.execution.messages cimport QueryOrder
from bomber.execution.messages cimport SubmitOrder
from bomber.execution.messages cimport SubmitOrderList
from bomber.model.data cimport Bar
from bomber.model.data cimport InstrumentClose
from bomber.model.data cimport InstrumentStatus
from bomber.model.data cimport OrderBookDelta
from bomber.model.data cimport OrderBookDeltas
from bomber.model.data cimport QuoteTick
from bomber.model.data cimport TradeTick
from bomber.model.events.account cimport AccountState
from bomber.model.events.order cimport OrderAccepted
from bomber.model.events.order cimport OrderCanceled
from bomber.model.events.order cimport OrderCancelRejected
from bomber.model.events.order cimport OrderDenied
from bomber.model.events.order cimport OrderEmulated
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
from bomber.model.events.position cimport PositionOpened
from bomber.model.instruments.betting cimport BettingInstrument
from bomber.model.instruments.binary_option cimport BinaryOption
from bomber.model.instruments.cfd cimport Cfd
from bomber.model.instruments.commodity cimport Commodity
from bomber.model.instruments.crypto_future cimport CryptoFuture
from bomber.model.instruments.crypto_futures_spread cimport CryptoFuturesSpread
from bomber.model.instruments.crypto_option cimport CryptoOption
from bomber.model.instruments.crypto_option_spread cimport CryptoOptionSpread
from bomber.model.instruments.crypto_perpetual cimport CryptoPerpetual
from bomber.model.instruments.currency_pair cimport CurrencyPair
from bomber.model.instruments.equity cimport Equity
from bomber.model.instruments.futures_contract cimport FuturesContract
from bomber.model.instruments.futures_spread cimport FuturesSpread
from bomber.model.instruments.index cimport IndexInstrument
from bomber.model.instruments.option_contract cimport OptionContract
from bomber.model.instruments.option_spread cimport OptionSpread
from bomber.model.instruments.perpetual_contract cimport PerpetualContract
from bomber.model.instruments.synthetic cimport SyntheticInstrument
from bomber.model.instruments.tokenized_asset cimport TokenizedAsset

from bomber.execution.reports import ExecutionMassStatus
from bomber.execution.reports import FillReport
from bomber.execution.reports import OrderStatusReport
from bomber.execution.reports import PositionStatusReport


# Default mappings for Nautilus objects
_OBJECT_TO_DICT_MAP: dict[str, Callable[[None], dict]] = {
    CancelOrder.__name__: CancelOrder.to_dict_c,
    CancelAllOrders.__name__: CancelAllOrders.to_dict_c,
    BatchCancelOrders.__name__: BatchCancelOrders.to_dict_c,
    GenerateFillReports.__name__: GenerateFillReports.to_dict_c,
    GenerateOrderStatusReport.__name__: GenerateOrderStatusReport.to_dict_c,
    GenerateOrderStatusReports.__name__: GenerateOrderStatusReports.to_dict_c,
    GeneratePositionStatusReports.__name__: GeneratePositionStatusReports.to_dict_c,
    GenerateExecutionMassStatus.__name__: GenerateExecutionMassStatus.to_dict_c,
    SubmitOrder.__name__: SubmitOrder.to_dict_c,
    SubmitOrderList.__name__: SubmitOrderList.to_dict_c,
    ModifyOrder.__name__: ModifyOrder.to_dict_c,
    QueryAccount.__name__: QueryAccount.to_dict_c,
    QueryOrder.__name__: QueryOrder.to_dict_c,
    ShutdownSystem.__name__: ShutdownSystem.to_dict_c,
    ComponentStateChanged.__name__: ComponentStateChanged.to_dict_c,
    TradingStateChanged.__name__: TradingStateChanged.to_dict_c,
    AccountState.__name__: AccountState.to_dict_c,
    OrderAccepted.__name__: OrderAccepted.to_dict_c,
    OrderCancelRejected.__name__: OrderCancelRejected.to_dict_c,
    OrderCanceled.__name__: OrderCanceled.to_dict_c,
    OrderDenied.__name__: OrderDenied.to_dict_c,
    OrderEmulated.__name__: OrderEmulated.to_dict_c,
    OrderExpired.__name__: OrderExpired.to_dict_c,
    OrderFilled.__name__: OrderFilled.to_dict_c,
    OrderInitialized.__name__: OrderInitialized.to_dict_c,
    OrderPendingCancel.__name__: OrderPendingCancel.to_dict_c,
    OrderPendingUpdate.__name__: OrderPendingUpdate.to_dict_c,
    OrderRejected.__name__: OrderRejected.to_dict_c,
    OrderReleased.__name__: OrderReleased.to_dict_c,
    OrderSubmitted.__name__: OrderSubmitted.to_dict_c,
    OrderTriggered.__name__: OrderTriggered.to_dict_c,
    OrderModifyRejected.__name__: OrderModifyRejected.to_dict_c,
    OrderUpdated.__name__: OrderUpdated.to_dict_c,
    PositionOpened.__name__: PositionOpened.to_dict_c,
    PositionChanged.__name__: PositionChanged.to_dict_c,
    PositionClosed.__name__: PositionClosed.to_dict_c,
    SyntheticInstrument.__name__: SyntheticInstrument.to_dict_c,
    BettingInstrument.__name__: BettingInstrument.to_dict_c,
    BinaryOption.__name__: BinaryOption.to_dict_c,
    Cfd.__name__: Cfd.to_dict_c,
    Commodity.__name__: Commodity.to_dict_c,
    CryptoFuture.__name__: CryptoFuture.to_dict_c,
    CryptoFuturesSpread.__name__: CryptoFuturesSpread.to_dict_c,
    CryptoOption.__name__: CryptoOption.to_dict_c,
    CryptoOptionSpread.__name__: CryptoOptionSpread.to_dict_c,
    CryptoPerpetual.__name__: CryptoPerpetual.to_dict_c,
    CurrencyPair.__name__: CurrencyPair.to_dict_c,
    Equity.__name__: Equity.to_dict_c,
    FuturesContract.__name__: FuturesContract.to_dict_c,
    FuturesSpread.__name__: FuturesSpread.to_dict_c,
    IndexInstrument.__name__: IndexInstrument.to_dict_c,
    OptionContract.__name__: OptionContract.to_dict_c,
    OptionSpread.__name__: OptionSpread.to_dict_c,
    PerpetualContract.__name__: PerpetualContract.to_dict_c,
    TokenizedAsset.__name__: TokenizedAsset.to_dict_c,
    OrderBookDelta.__name__: OrderBookDelta.to_dict_c,
    OrderBookDeltas.__name__: OrderBookDeltas.to_dict_c,
    TradeTick.__name__: TradeTick.to_dict_c,
    QuoteTick.__name__: QuoteTick.to_dict_c,
    Bar.__name__: Bar.to_dict_c,
    InstrumentStatus.__name__: InstrumentStatus.to_dict_c,
    InstrumentClose.__name__: InstrumentClose.to_dict_c,
    OrderStatusReport.__name__: OrderStatusReport.to_dict,
    FillReport.__name__: FillReport.to_dict,
    PositionStatusReport.__name__: PositionStatusReport.to_dict,
    ExecutionMassStatus.__name__: ExecutionMassStatus.to_dict,
}


# Default mappings for Nautilus objects
_OBJECT_FROM_DICT_MAP: dict[str, Callable[[dict], Any]] = {
    CancelOrder.__name__: CancelOrder.from_dict_c,
    CancelAllOrders.__name__: CancelAllOrders.from_dict_c,
    BatchCancelOrders.__name__: BatchCancelOrders.from_dict_c,
    GenerateFillReports.__name__: GenerateFillReports.from_dict_c,
    GenerateOrderStatusReport.__name__: GenerateOrderStatusReport.from_dict_c,
    GenerateOrderStatusReports.__name__: GenerateOrderStatusReports.from_dict_c,
    GeneratePositionStatusReports.__name__: GeneratePositionStatusReports.from_dict_c,
    GenerateExecutionMassStatus.__name__: GenerateExecutionMassStatus.from_dict_c,
    SubmitOrder.__name__: SubmitOrder.from_dict_c,
    SubmitOrderList.__name__: SubmitOrderList.from_dict_c,
    ModifyOrder.__name__: ModifyOrder.from_dict_c,
    QueryAccount.__name__: QueryAccount.from_dict_c,
    QueryOrder.__name__: QueryOrder.from_dict_c,
    ShutdownSystem.__name__: ShutdownSystem.from_dict_c,
    ComponentStateChanged.__name__: ComponentStateChanged.from_dict_c,
    TradingStateChanged.__name__: TradingStateChanged.from_dict_c,
    AccountState.__name__: AccountState.from_dict_c,
    OrderAccepted.__name__: OrderAccepted.from_dict_c,
    OrderCancelRejected.__name__: OrderCancelRejected.from_dict_c,
    OrderCanceled.__name__: OrderCanceled.from_dict_c,
    OrderDenied.__name__: OrderDenied.from_dict_c,
    OrderEmulated.__name__: OrderEmulated.from_dict_c,
    OrderExpired.__name__: OrderExpired.from_dict_c,
    OrderFilled.__name__: OrderFilled.from_dict_c,
    OrderInitialized.__name__: OrderInitialized.from_dict_c,
    OrderPendingCancel.__name__: OrderPendingCancel.from_dict_c,
    OrderPendingUpdate.__name__: OrderPendingUpdate.from_dict_c,
    OrderReleased.__name__: OrderReleased.from_dict_c,
    OrderRejected.__name__: OrderRejected.from_dict_c,
    OrderSubmitted.__name__: OrderSubmitted.from_dict_c,
    OrderTriggered.__name__: OrderTriggered.from_dict_c,
    OrderModifyRejected.__name__: OrderModifyRejected.from_dict_c,
    OrderUpdated.__name__: OrderUpdated.from_dict_c,
    PositionOpened.__name__: PositionOpened.from_dict_c,
    PositionChanged.__name__: PositionChanged.from_dict_c,
    PositionClosed.__name__: PositionClosed.from_dict_c,
    SyntheticInstrument.__name__: SyntheticInstrument.from_dict_c,
    BettingInstrument.__name__: BettingInstrument.from_dict_c,
    BinaryOption.__name__: BinaryOption.from_dict_c,
    Cfd.__name__: Cfd.from_dict_c,
    Commodity.__name__: Commodity.from_dict_c,
    CryptoFuture.__name__: CryptoFuture.from_dict_c,
    CryptoFuturesSpread.__name__: CryptoFuturesSpread.from_dict_c,
    CryptoOption.__name__: CryptoOption.from_dict_c,
    CryptoOptionSpread.__name__: CryptoOptionSpread.from_dict_c,
    CryptoPerpetual.__name__: CryptoPerpetual.from_dict_c,
    CurrencyPair.__name__: CurrencyPair.from_dict_c,
    Equity.__name__: Equity.from_dict_c,
    FuturesContract.__name__: FuturesContract.from_dict_c,
    FuturesSpread.__name__: FuturesSpread.from_dict_c,
    IndexInstrument.__name__: IndexInstrument.from_dict_c,
    OptionContract.__name__: OptionContract.from_dict_c,
    OptionSpread.__name__: OptionSpread.from_dict_c,
    PerpetualContract.__name__: PerpetualContract.from_dict_c,
    TokenizedAsset.__name__: TokenizedAsset.from_dict_c,
    OrderBookDelta.__name__: OrderBookDelta.from_dict_c,
    OrderBookDeltas.__name__: OrderBookDeltas.from_dict_c,
    TradeTick.__name__: TradeTick.from_dict_c,
    QuoteTick.__name__: QuoteTick.from_dict_c,
    Bar.__name__: Bar.from_dict_c,
    InstrumentStatus.__name__: InstrumentStatus.from_dict_c,
    InstrumentClose.__name__: InstrumentClose.from_dict_c,
    OrderStatusReport.__name__: OrderStatusReport.from_dict,
    FillReport.__name__: FillReport.from_dict,
    PositionStatusReport.__name__: PositionStatusReport.from_dict,
    ExecutionMassStatus.__name__: ExecutionMassStatus.from_dict,
}


_EXTERNAL_PUBLISHABLE_TYPES = {
    str,
    int,
    float,
    bytes,
    SubmitOrder,
    SubmitOrderList,
    ModifyOrder,
    CancelOrder,
    CancelAllOrders,
    BatchCancelOrders,
    GenerateFillReports,
    GenerateOrderStatusReport,
    GenerateOrderStatusReports,
    GeneratePositionStatusReports,
    GenerateExecutionMassStatus,
    QueryAccount,
    QueryOrder,
    ShutdownSystem,
    ComponentStateChanged,
    TradingStateChanged,
    AccountState,
    OrderAccepted,
    OrderCancelRejected,
    OrderCanceled,
    OrderDenied,
    OrderEmulated,
    OrderExpired,
    OrderFilled,
    OrderInitialized,
    OrderPendingCancel,
    OrderPendingUpdate,
    OrderReleased,
    OrderRejected,
    OrderSubmitted,
    OrderTriggered,
    OrderModifyRejected,
    OrderUpdated,
    PositionOpened,
    PositionChanged,
    PositionClosed,
    SyntheticInstrument,
    BettingInstrument,
    BinaryOption,
    Cfd,
    Commodity,
    CryptoFuture,
    CryptoFuturesSpread,
    CryptoOption,
    CryptoOptionSpread,
    CryptoPerpetual,
    CurrencyPair,
    Equity,
    FuturesContract,
    FuturesSpread,
    IndexInstrument,
    OptionContract,
    OptionSpread,
    PerpetualContract,
    OrderBookDelta,
    OrderBookDeltas,
    TradeTick,
    QuoteTick,
    Bar,
    InstrumentStatus,
    InstrumentClose,
    OrderStatusReport,
    FillReport,
    PositionStatusReport,
    ExecutionMassStatus,
}


cpdef void register_serializable_type(
    cls: type,
    to_dict: Callable[[Any], dict[str, Any]],
    from_dict: Callable[[dict[str, Any]], Any],
):
    """
    Register the given type with the global serialization type maps.

    The `type` will also be registered as an external publishable type and
    will be published externally on the message bus unless also added to
    the `MessageBusConfig.types_filter`.

    Parameters
    ----------
    cls : type
        The type to register.
    to_dict : Callable[[Any], dict[str, Any]]
        The delegate to instantiate a dict of primitive types from an object.
    from_dict : Callable[[dict[str, Any]], Any]
        The delegate to instantiate an object from a dict of primitive types.

    Raises
    ------
    TypeError
        If `to_dict` or `from_dict` are not of type `Callable`.
    KeyError
        If `type` already registered with the global type maps.

    """
    Condition.callable(to_dict, "to_dict")
    Condition.callable(from_dict, "from_dict")
    Condition.not_in(cls.__name__, _OBJECT_TO_DICT_MAP, "cls.__name__", "_OBJECT_TO_DICT_MAP")
    Condition.not_in(cls.__name__, _OBJECT_FROM_DICT_MAP, "cls.__name__", "_OBJECT_FROM_DICT_MAP")

    _OBJECT_TO_DICT_MAP[cls.__name__] = to_dict
    _OBJECT_FROM_DICT_MAP[cls.__name__] = from_dict
    _EXTERNAL_PUBLISHABLE_TYPES.add(cls)


cdef class Serializer:
    """
    The base class for all serializers.

    Warnings
    --------
    This class should not be used directly, but through a concrete subclass.
    """

    def __init__(self):
        super().__init__()

    cpdef bytes serialize(self, object obj):
        """Abstract method (implement in subclass)."""
        raise NotImplementedError("method `serialize` must be implemented in the subclass")  # pragma: no cover

    cpdef object deserialize(self, bytes obj_bytes):
        """Abstract method (implement in subclass)."""
        raise NotImplementedError("method `deserialize` must be implemented in the subclass")  # pragma: no cover
