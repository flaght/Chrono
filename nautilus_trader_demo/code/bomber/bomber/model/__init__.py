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
The `model` subpackage defines a rich trading domain model.

The domain model is agnostic of any system design, seeking to represent the logic and
state transitions of trading in a generic way. Many system implementations could be
built around this domain model.

"""

from decimal import ROUND_HALF_UP
from decimal import Decimal

from bomber.core import bomber_pyo3
from bomber.model.book import BookLevel
from bomber.model.book import OrderBook
from bomber.model.data import Bar
from bomber.model.data import BarSpecification
from bomber.model.data import BarType
from bomber.model.data import BookOrder
from bomber.model.data import CustomData
from bomber.model.data import DataType
from bomber.model.data import FundingRateUpdate
from bomber.model.data import InstrumentClose
from bomber.model.data import InstrumentStatus
from bomber.model.data import MarkPriceUpdate
from bomber.model.data import OrderBookDelta
from bomber.model.data import OrderBookDeltas
from bomber.model.data import OrderBookDepth10
from bomber.model.data import QuoteTick
from bomber.model.data import TradeTick
from bomber.model.identifiers import AccountId
from bomber.model.identifiers import ClientId
from bomber.model.identifiers import ClientOrderId
from bomber.model.identifiers import ComponentId
from bomber.model.identifiers import ExecAlgorithmId
from bomber.model.identifiers import InstrumentId
from bomber.model.identifiers import OrderListId
from bomber.model.identifiers import PositionId
from bomber.model.identifiers import StrategyId
from bomber.model.identifiers import Symbol
from bomber.model.identifiers import TradeId
from bomber.model.identifiers import TraderId
from bomber.model.identifiers import Venue
from bomber.model.identifiers import VenueOrderId
from bomber.model.objects import FIXED_PRECISION
from bomber.model.objects import AccountBalance
from bomber.model.objects import Currency
from bomber.model.objects import MarginBalance
from bomber.model.objects import Money
from bomber.model.objects import Price
from bomber.model.objects import Quantity
from bomber.model.position import Position


# Defines all order book data types (capable of updating an L2_MBP and L3_MBO book)
BOOK_DATA_TYPES: set[type] = {
    OrderBookDelta,
    OrderBookDeltas,
    OrderBookDepth10,
}

NAUTILUS_PYO3_DATA_TYPES: tuple[type, ...] = (
    bomber_pyo3.OrderBookDelta,
    bomber_pyo3.OrderBookDepth10,
    bomber_pyo3.QuoteTick,
    bomber_pyo3.TradeTick,
    bomber_pyo3.Bar,
)


# Convert the given value into the raw integer representation based on the given precision
# and currently compiled precision mode (128-bit for HIGH_PRECISION or 64-bit).
def convert_to_raw_int(value, precision: int) -> int:
    # Use Decimal for exact decimal arithmetic to avoid platform-specific
    # floating-point rounding differences.
    decimal_value = Decimal(str(value))
    quantized = decimal_value.quantize(Decimal(10) ** -precision, rounding=ROUND_HALF_UP)
    return int(quantized * (10**FIXED_PRECISION))


__all__ = [
    "AccountBalance",
    "AccountId",
    "Bar",
    "BarSpecification",
    "BarType",
    "BookLevel",
    "BookOrder",
    "ClientId",
    "ClientOrderId",
    "ComponentId",
    "Currency",
    "CustomData",
    "DataType",
    "ExecAlgorithmId",
    "FundingRateUpdate",
    "InstrumentClose",
    "InstrumentId",
    "InstrumentStatus",
    "MarginBalance",
    "MarkPriceUpdate",
    "Money",
    "OrderBook",
    "OrderBookDelta",
    "OrderBookDeltas",
    "OrderBookDepth10",
    "OrderListId",
    "Position",
    "PositionId",
    "Price",
    "Quantity",
    "QuoteTick",
    "StrategyId",
    "Symbol",
    "TradeId",
    "TradeTick",
    "TraderId",
    "Venue",
    "VenueOrderId",
]
