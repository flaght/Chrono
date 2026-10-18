"""
z001_core.py — 量能时钟均匀度因子（Volume-Time Uniformity Factor）

核心思想：
将物理时间轴重新参数化为成交量时间（事件时间），度量信息在成交量维度上的均匀程度。
若信息均匀到达，每个量能桶内的已实现波动率应大致相等；若信息集中或稀疏，则变异系数增大。
通过与物理时间下的均匀度对比，剥离物理时间结构的影响，提供与价格波动率正交的增量信息。

因子公式：
    metric_d = CV_voltime,d - CV_physical,d
    F_t      = mean(metric_d) over lookback days

本实现采用滚动窗口近似：
- 物理时间桶：每个交易日等分为 K_phys 个连续时间桶，桶内收益平方和作为桶 RV；
- 量能时间桶：以单位成交量 RV（桶 RV / 桶成交量）近似量能均匀度；
- 日度变异系数：在 trailing day_window 分钟上计算 std/mean；
- 因子聚合：在 trailing weriod * bars_per_day 分钟上取均值。
"""
import pdb
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def _infer_bars_per_day(idx, default=240):
    """从 DatetimeIndex 推断每日 bar 数（稳健中位数，剔除首尾不完整日影响）。"""
    try:
        days = idx.floor('D')
        day_changed = np.zeros(len(idx), dtype=bool)
        day_changed[1:] = days[1:] != days[:-1]
        day_starts = np.where(day_changed)[0]
        if len(day_starts) == 0:
            return default
        boundaries = np.concatenate(([0], day_starts, [len(idx)]))
        lengths = np.diff(boundaries)
        return max(1, int(np.median(lengths)))
    except Exception:
        return default


def z001(close, openint, window, weriod, ewm=False,
         bars_per_day=None, volume_buckets=-1, physical_buckets=-1):
    """
    量能时钟均匀度因子。

    Parameters
    ----------
    close, volume : DataFrame
        宽表行情数据，行索引为 trade_time，列索引为 code。
    window : int
        最终平滑窗口（框架强制要求，仅用于 return 前平滑）。
    weriod : int
        滚动窗口天数（lookback_days，默认 20）。
    ewm : bool
        是否使用 EWM 平滑。
    bars_per_day : int, optional
        每日分钟 bar 数。None 时从时间索引自动推断。
    volume_buckets : int
        量能桶数，-1 表示自适应取 floor(sqrt(bars_per_day))。
    physical_buckets : int
        物理时间桶数，-1 表示与 volume_buckets 相同。
    min_bars_per_day : int
        每日最少分钟数（保留参数，滚动计算通过 min_periods 隐式控制）。
    """
    method = 'ewm' if ewm else 'rolling'
    # ---- 参数预处理 ----
    if bars_per_day is None or bars_per_day <= 0:
        bars_per_day = _infer_bars_per_day(close.index, default=240)
    bars_per_day = max(1, int(bars_per_day))

    if volume_buckets is None or volume_buckets <= 0:
        K_vol = max(1, int(np.floor(np.sqrt(bars_per_day))))
    else:
        K_vol = max(1, int(volume_buckets))

    if physical_buckets is None or physical_buckets <= 0:
        K_phys = K_vol
    else:
        K_phys = max(1, int(physical_buckets))

    phys_bucket_len = max(1, bars_per_day // K_phys)
    vol_bucket_len = max(1, bars_per_day // K_vol)

    day_window = bars_per_day
    total_window = max(1, weriod * bars_per_day)

    # ---- 1. 分钟对数收益率 ----
    rets = safe_shift(close, 1)
    rets_sq = rets ** 2

    # ---- 2. 物理时间桶已实现波动率 ----
    # 每个物理时间桶（连续 phys_bucket_len 分钟）的收益平方和
    rv_phys_bucket = roller_sum(rets_sq, phys_bucket_len, 1, method)
    # 日度变异系数：桶 RV 的标准差 / 桶 RV 的均值
    mean_phys_day = roller_mean(rv_phys_bucket, day_window, 1, method)
    std_phys_day = roller_std(rv_phys_bucket, day_window, 1, method)
    cv_phys_day = safe_div(std_phys_day, mean_phys_day)

    # ---- 3. 量能时间桶已实现波动率 ----
    # 近似：每个量能物理桶内的单位成交量 RV
    # 若 RV 与成交量成正比，则单位成交量 RV 恒定 -> 量能均匀度最高
    rv_vol_bucket = roller_sum(rets_sq, vol_bucket_len, 1, method)
    vol_vol_bucket = roller_sum(openint, vol_bucket_len, 1, method)
    rv_per_vol_bucket = safe_div(rv_vol_bucket, vol_vol_bucket)
    mean_vol_day = roller_mean(rv_per_vol_bucket, day_window, 1, method)
    std_vol_day = roller_std(rv_per_vol_bucket, day_window, 1, method)
    cv_vol_day = safe_div(std_vol_day, mean_vol_day)

    # ---- 4. 日度均匀度差异 ----
    metric_day = cv_vol_day - cv_phys_day

    # ---- 5. 因子聚合：在 weriod 天窗口上取均值 ----
    alpha_raw = roller_mean(metric_day, total_window, 1, method)

    # ---- 6. 框架强制最终平滑 ----
    alpha = roller_mean(alpha_raw, window, 1, method)

    return alpha