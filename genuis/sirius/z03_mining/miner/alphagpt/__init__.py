"""由 Transformer 策略生成公式、使用 Polars 执行的因子挖掘器。"""

from .launcher import Launcher
from .config import AlphaGPTConfig

__all__ = ["Launcher", "AlphaGPTConfig"]
