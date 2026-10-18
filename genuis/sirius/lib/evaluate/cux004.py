"""cux004: 信号专用综合评估器.

直接继承并复用 lib.evaluate.cux001.FactorEvaluate1 的全部 Polars 高性能实现与实盘信号诊断能力。
"""

try:
    from lib.evaluate.cux001 import FactorEvaluate1, FactorEvaluatePolars
except ImportError:
    from .cux001 import FactorEvaluate1, FactorEvaluatePolars

__all__ = ["FactorEvaluate1", "FactorEvaluatePolars"]
