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
The `analysis` subpackage groups components relating to trading performance statistics
and analysis.
"""

from bomber.analysis.analyzer import PortfolioAnalyzer
from bomber.analysis.config import GridLayout
from bomber.analysis.config import TearsheetBarsWithFillsChart
from bomber.analysis.config import TearsheetChart
from bomber.analysis.config import TearsheetConfig
from bomber.analysis.config import TearsheetCustomChart
from bomber.analysis.config import TearsheetDistributionChart
from bomber.analysis.config import TearsheetDrawdownChart
from bomber.analysis.config import TearsheetEquityChart
from bomber.analysis.config import TearsheetMonthlyReturnsChart
from bomber.analysis.config import TearsheetRollingSharpeChart
from bomber.analysis.config import TearsheetRunInfoChart
from bomber.analysis.config import TearsheetStatsTableChart
from bomber.analysis.config import TearsheetYearlyReturnsChart
from bomber.analysis.reporter import ReportProvider
from bomber.analysis.statistic import PortfolioStatistic
from bomber.analysis.tearsheet import create_drawdown_chart
from bomber.analysis.tearsheet import create_equity_curve
from bomber.analysis.tearsheet import create_monthly_returns_heatmap
from bomber.analysis.tearsheet import create_returns_distribution
from bomber.analysis.tearsheet import create_rolling_sharpe
from bomber.analysis.tearsheet import create_tearsheet
from bomber.analysis.tearsheet import create_tearsheet_from_stats
from bomber.analysis.tearsheet import create_yearly_returns
from bomber.analysis.tearsheet import get_chart
from bomber.analysis.tearsheet import list_charts
from bomber.analysis.tearsheet import register_chart
from bomber.analysis.themes import get_theme
from bomber.analysis.themes import list_themes
from bomber.analysis.themes import register_theme
from bomber.core.bomber_pyo3 import CAGR
from bomber.core.bomber_pyo3 import Alpha
from bomber.core.bomber_pyo3 import AvgLoser
from bomber.core.bomber_pyo3 import AvgWinner
from bomber.core.bomber_pyo3 import BetaRatio
from bomber.core.bomber_pyo3 import CalmarRatio
from bomber.core.bomber_pyo3 import Expectancy
from bomber.core.bomber_pyo3 import InformationRatio
from bomber.core.bomber_pyo3 import LongRatio
from bomber.core.bomber_pyo3 import MaxDrawdown
from bomber.core.bomber_pyo3 import MaxLoser
from bomber.core.bomber_pyo3 import MaxWinner
from bomber.core.bomber_pyo3 import MinLoser
from bomber.core.bomber_pyo3 import MinWinner
from bomber.core.bomber_pyo3 import ProfitFactor
from bomber.core.bomber_pyo3 import ReturnsAverage
from bomber.core.bomber_pyo3 import ReturnsAverageLoss
from bomber.core.bomber_pyo3 import ReturnsAverageWin
from bomber.core.bomber_pyo3 import ReturnsVolatility
from bomber.core.bomber_pyo3 import RiskReturnRatio
from bomber.core.bomber_pyo3 import SharpeRatio
from bomber.core.bomber_pyo3 import SortinoRatio
from bomber.core.bomber_pyo3 import TrackingError
from bomber.core.bomber_pyo3 import TreynorRatio
from bomber.core.bomber_pyo3 import WinRate


__all__ = [
    "CAGR",
    "Alpha",
    "AvgLoser",
    "AvgWinner",
    "BetaRatio",
    "CalmarRatio",
    "Expectancy",
    "GridLayout",
    "InformationRatio",
    "LongRatio",
    "MaxDrawdown",
    "MaxLoser",
    "MaxWinner",
    "MinLoser",
    "MinWinner",
    "PortfolioAnalyzer",
    "PortfolioStatistic",
    "ProfitFactor",
    "ReportProvider",
    "ReturnsAverage",
    "ReturnsAverageLoss",
    "ReturnsAverageWin",
    "ReturnsVolatility",
    "RiskReturnRatio",
    "SharpeRatio",
    "SortinoRatio",
    "TearsheetBarsWithFillsChart",
    "TearsheetChart",
    "TearsheetConfig",
    "TearsheetCustomChart",
    "TearsheetDistributionChart",
    "TearsheetDrawdownChart",
    "TearsheetEquityChart",
    "TearsheetMonthlyReturnsChart",
    "TearsheetRollingSharpeChart",
    "TearsheetRunInfoChart",
    "TearsheetStatsTableChart",
    "TearsheetYearlyReturnsChart",
    "TrackingError",
    "TreynorRatio",
    "WinRate",
    "create_drawdown_chart",
    "create_equity_curve",
    "create_monthly_returns_heatmap",
    "create_returns_distribution",
    "create_rolling_sharpe",
    "create_tearsheet",
    "create_tearsheet_from_stats",
    "create_yearly_returns",
    "get_chart",
    "get_theme",
    "list_charts",
    "list_themes",
    "register_chart",
    "register_theme",
]
