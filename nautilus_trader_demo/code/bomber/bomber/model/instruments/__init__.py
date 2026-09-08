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
Defines tradable asset/contract instruments with specific properties dependent on the
asset class and instrument class.
"""

from bomber.model.instruments.base import Instrument
from bomber.model.instruments.base import instruments_from_pyo3
from bomber.model.instruments.betting import BettingInstrument
from bomber.model.instruments.binary_option import BinaryOption
from bomber.model.instruments.cfd import Cfd
from bomber.model.instruments.commodity import Commodity
from bomber.model.instruments.crypto_future import CryptoFuture
from bomber.model.instruments.crypto_futures_spread import CryptoFuturesSpread
from bomber.model.instruments.crypto_option import CryptoOption
from bomber.model.instruments.crypto_option_spread import CryptoOptionSpread
from bomber.model.instruments.crypto_perpetual import CryptoPerpetual
from bomber.model.instruments.currency_pair import CurrencyPair
from bomber.model.instruments.equity import Equity
from bomber.model.instruments.futures_contract import FuturesContract
from bomber.model.instruments.futures_spread import FuturesSpread
from bomber.model.instruments.index import IndexInstrument
from bomber.model.instruments.option_contract import OptionContract
from bomber.model.instruments.option_spread import OptionSpread
from bomber.model.instruments.perpetual_contract import PerpetualContract
from bomber.model.instruments.synthetic import SyntheticInstrument
from bomber.model.instruments.tokenized_asset import TokenizedAsset


__all__ = [
    "BettingInstrument",
    "BinaryOption",
    "Cfd",
    "Commodity",
    "CryptoFuture",
    "CryptoFuturesSpread",
    "CryptoOption",
    "CryptoOptionSpread",
    "CryptoPerpetual",
    "CurrencyPair",
    "Equity",
    "FuturesContract",
    "FuturesSpread",
    "IndexInstrument",
    "Instrument",
    "OptionContract",
    "OptionSpread",
    "PerpetualContract",
    "SyntheticInstrument",
    "TokenizedAsset",
    "instruments_from_pyo3",
]
