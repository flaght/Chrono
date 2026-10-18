"""兼容旧导入；直接在线运行时位于runtime/live/runtime.py。"""

from .live.runtime import DirectLiveRuntime

__all__ = ["DirectLiveRuntime"]
