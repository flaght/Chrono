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
The `config` subpackage groups all configuration classes and factories.

All configurations inherit from :class:`NautilusConfig` which in turn inherits from :class:`msgspec.Struct`.

"""

from bomber.backtest.config import BacktestDataConfig
from bomber.backtest.config import BacktestEngineConfig
from bomber.backtest.config import BacktestRunConfig
from bomber.backtest.config import BacktestVenueConfig
from bomber.backtest.config import FeeModelFactory
from bomber.backtest.config import FillModelConfig
from bomber.backtest.config import FillModelFactory
from bomber.backtest.config import FixedFeeModelConfig
from bomber.backtest.config import FXRolloverInterestConfig
from bomber.backtest.config import ImportableFeeModelConfig
from bomber.backtest.config import ImportableFillModelConfig
from bomber.backtest.config import ImportableLatencyModelConfig
from bomber.backtest.config import LatencyModelConfig
from bomber.backtest.config import LatencyModelFactory
from bomber.backtest.config import MakerTakerFeeModelConfig
from bomber.backtest.config import PerContractFeeModelConfig
from bomber.backtest.config import SimulationModuleConfig
from bomber.cache.config import CacheConfig
from bomber.common.config import ActorConfig
from bomber.common.config import ActorFactory
from bomber.common.config import DatabaseConfig
from bomber.common.config import ImportableActorConfig
from bomber.common.config import ImportableConfig
from bomber.common.config import InstrumentProviderConfig
from bomber.common.config import InvalidConfiguration
from bomber.common.config import LoggingConfig
from bomber.common.config import MessageBusConfig
from bomber.common.config import NautilusConfig
from bomber.common.config import NonNegativeFloat
from bomber.common.config import NonNegativeInt
from bomber.common.config import OrderEmulatorConfig
from bomber.common.config import PositiveFloat
from bomber.common.config import PositiveInt
from bomber.common.config import msgspec_decoding_hook
from bomber.common.config import msgspec_encoding_hook
from bomber.common.config import register_config_decoding
from bomber.common.config import register_config_encoding
from bomber.common.config import resolve_config_path
from bomber.common.config import resolve_path
from bomber.common.config import tokenize_config
from bomber.data.config import DataEngineConfig
from bomber.execution.config import ExecAlgorithmConfig
from bomber.execution.config import ExecAlgorithmFactory
from bomber.execution.config import ExecEngineConfig
from bomber.execution.config import ImportableExecAlgorithmConfig
from bomber.live.config import ControllerConfig
from bomber.live.config import ControllerFactory
from bomber.live.config import ImportableControllerConfig
from bomber.live.config import LiveDataClientConfig
from bomber.live.config import LiveDataEngineConfig
from bomber.live.config import LiveExecClientConfig
from bomber.live.config import LiveExecEngineConfig
from bomber.live.config import LiveRiskEngineConfig
from bomber.live.config import RoutingConfig
from bomber.live.config import TradingNodeConfig
from bomber.persistence.config import DataCatalogConfig
from bomber.persistence.config import StreamingConfig
from bomber.portfolio.config import PortfolioConfig
from bomber.risk.config import RiskEngineConfig
from bomber.system.config import NautilusKernelConfig
from bomber.trading.config import ImportableStrategyConfig
from bomber.trading.config import StrategyConfig
from bomber.trading.config import StrategyFactory


__all__ = [
    "ActorConfig",
    "ActorFactory",
    "BacktestDataConfig",
    "BacktestEngineConfig",
    "BacktestRunConfig",
    "BacktestVenueConfig",
    "CacheConfig",
    "ControllerConfig",
    "ControllerFactory",
    "DataCatalogConfig",
    "DataEngineConfig",
    "DatabaseConfig",
    "ExecAlgorithmConfig",
    "ExecAlgorithmFactory",
    "ExecEngineConfig",
    "FXRolloverInterestConfig",
    "FeeModelFactory",
    "FillModelConfig",
    "FillModelFactory",
    "FixedFeeModelConfig",
    "ImportableActorConfig",
    "ImportableConfig",
    "ImportableControllerConfig",
    "ImportableExecAlgorithmConfig",
    "ImportableFeeModelConfig",
    "ImportableFillModelConfig",
    "ImportableLatencyModelConfig",
    "ImportableStrategyConfig",
    "InstrumentProviderConfig",
    "InvalidConfiguration",
    "LatencyModelConfig",
    "LatencyModelFactory",
    "LiveDataClientConfig",
    "LiveDataEngineConfig",
    "LiveExecClientConfig",
    "LiveExecEngineConfig",
    "LiveRiskEngineConfig",
    "LoggingConfig",
    "MakerTakerFeeModelConfig",
    "MessageBusConfig",
    "NautilusConfig",
    "NautilusKernelConfig",
    "NonNegativeFloat",
    "NonNegativeInt",
    "OrderEmulatorConfig",
    "PerContractFeeModelConfig",
    "PortfolioConfig",
    "PositiveFloat",
    "PositiveInt",
    "RiskEngineConfig",
    "RoutingConfig",
    "SimulationModuleConfig",
    "StrategyConfig",
    "StrategyFactory",
    "StreamingConfig",
    "TradingNodeConfig",
    "msgspec_decoding_hook",
    "msgspec_encoding_hook",
    "register_config_decoding",
    "register_config_encoding",
    "resolve_config_path",
    "resolve_path",
    "tokenize_config",
]
