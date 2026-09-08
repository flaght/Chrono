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
The `indicator` subpackage provides a set of efficient indicators and analyzers.

These are classes which can be used for signal discovery and filtering. The idea is to
use the provided indicators as is, or as inspiration for a trader to implement their own
proprietary indicator algorithms with the platform.

"""

from bomber.indicators.averages import AdaptiveMovingAverage
from bomber.indicators.averages import DoubleExponentialMovingAverage
from bomber.indicators.averages import ExponentialMovingAverage
from bomber.indicators.averages import HullMovingAverage
from bomber.indicators.averages import MovingAverage
from bomber.indicators.averages import MovingAverageFactory
from bomber.indicators.averages import MovingAverageType
from bomber.indicators.averages import SimpleMovingAverage
from bomber.indicators.averages import VariableIndexDynamicAverage
from bomber.indicators.averages import WeightedMovingAverage
from bomber.indicators.averages import WilderMovingAverage
from bomber.indicators.base import Indicator
from bomber.indicators.fuzzy_candlesticks import FuzzyCandle
from bomber.indicators.fuzzy_candlesticks import FuzzyCandlesticks
from bomber.indicators.fuzzy_enums import CandleBodySize
from bomber.indicators.fuzzy_enums import CandleDirection
from bomber.indicators.fuzzy_enums import CandleSize
from bomber.indicators.fuzzy_enums import CandleWickSize
from bomber.indicators.momentum import ChandeMomentumOscillator
from bomber.indicators.momentum import CommodityChannelIndex
from bomber.indicators.momentum import EfficiencyRatio
from bomber.indicators.momentum import PsychologicalLine
from bomber.indicators.momentum import RateOfChange
from bomber.indicators.momentum import RelativeStrengthIndex
from bomber.indicators.momentum import RelativeVolatilityIndex
from bomber.indicators.momentum import Stochastics
from bomber.indicators.momentum import StochasticsDMethod
from bomber.indicators.spread_analyzer import SpreadAnalyzer
from bomber.indicators.trend import ArcherMovingAveragesTrends
from bomber.indicators.trend import AroonOscillator
from bomber.indicators.trend import Bias
from bomber.indicators.trend import DirectionalMovement
from bomber.indicators.trend import IchimokuCloud
from bomber.indicators.trend import LinearRegression
from bomber.indicators.trend import MovingAverageConvergenceDivergence
from bomber.indicators.trend import Swings
from bomber.indicators.volatility import AverageTrueRange
from bomber.indicators.volatility import BollingerBands
from bomber.indicators.volatility import DonchianChannel
from bomber.indicators.volatility import KeltnerChannel
from bomber.indicators.volatility import KeltnerPosition
from bomber.indicators.volatility import VerticalHorizontalFilter
from bomber.indicators.volatility import VolatilityRatio
from bomber.indicators.volume import KlingerVolumeOscillator
from bomber.indicators.volume import OnBalanceVolume
from bomber.indicators.volume import Pressure
from bomber.indicators.volume import VolumeWeightedAveragePrice


__all__ = [
    "AdaptiveMovingAverage",
    "ArcherMovingAveragesTrends",
    "AroonOscillator",
    "AverageTrueRange",
    "Bias",
    "BollingerBands",
    "CandleBodySize",
    "CandleDirection",
    "CandleSize",
    "CandleWickSize",
    "ChandeMomentumOscillator",
    "CommodityChannelIndex",
    "DirectionalMovement",
    "DonchianChannel",
    "DoubleExponentialMovingAverage",
    "EfficiencyRatio",
    "ExponentialMovingAverage",
    "FuzzyCandle",
    "FuzzyCandlesticks",
    "HullMovingAverage",
    "IchimokuCloud",
    "Indicator",
    "KeltnerChannel",
    "KeltnerPosition",
    "KlingerVolumeOscillator",
    "LinearRegression",
    "MovingAverage",
    "MovingAverageConvergenceDivergence",
    "MovingAverageFactory",
    "MovingAverageType",
    "OnBalanceVolume",
    "Pressure",
    "PsychologicalLine",
    "RateOfChange",
    "RelativeStrengthIndex",
    "RelativeVolatilityIndex",
    "SimpleMovingAverage",
    "SpreadAnalyzer",
    "Stochastics",
    "StochasticsDMethod",
    "Swings",
    "VariableIndexDynamicAverage",
    "VerticalHorizontalFilter",
    "VolatilityRatio",
    "VolumeWeightedAveragePrice",
    "WeightedMovingAverage",
    "WilderMovingAverage",
]
