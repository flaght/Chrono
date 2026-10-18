"""可扩展特征协议、版本缓存、对齐及一次性决策帧。"""
from .contracts import (
    Alignment, CommoditySpot, DatasetRegistry, DatasetSpec, ExternalDataRecord,
    FeatureError, FeatureQuery, FeatureRequirement, FeatureUnavailable,
    FrequencySpec, IndexPrice, MinuteSignal, PayloadField,
)
from .store import FeatureStore
from .alignment import FeatureHub, FeatureSnapshot
from .frames import FeatureFrameAssembler, FrameOutcome, StrategyFrame
from .replay import MarketFrameInput, replay_feature_frames
from .schema import dataset_from_definition

__all__ = [
    "Alignment", "CommoditySpot", "DatasetRegistry", "DatasetSpec", "ExternalDataRecord",
    "FeatureError", "FeatureQuery", "FeatureRequirement", "FeatureUnavailable", "FrequencySpec",
    "IndexPrice", "MinuteSignal", "PayloadField", "FeatureStore", "FeatureHub", "FeatureSnapshot",
    "FeatureFrameAssembler", "FrameOutcome", "StrategyFrame",
    "MarketFrameInput", "replay_feature_frames",
    "dataset_from_definition",
]
