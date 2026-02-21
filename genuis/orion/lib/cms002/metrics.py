# -*- encoding:utf-8 -*-
"""
Cython 加速版 ArbMetrics 评估器

架构：
    metrics.py (Python 层) — 负责 pandas ↔ numpy 转换 + 高层逻辑
    booster.pyx (Cython 层) — 负责核心数值计算
"""
import pdb
import functools
import numpy as np
import pandas as pd
from collections import namedtuple
from .booster import Booster

DALIY_PER_YEAR = 252
WEEKLY_PER_YEAR = 52
MONTHLY_PER_YEAR = 12
QUARTERLY_PER_YEAR = 4
YEARLY_PER_YEAR = 1
HOURLY_PER_YEAR = 365 * 24  # 8760

OLNY_LONG = 'long'
OLNY_SHORT = 'short'
BOTH_SIDE = 'both'
TOP_N = 'topn'

POSITIVE = 1
NEGATIVE = -1

EXCESS = 1
ABSOLUTE = -1


class EvaluateTuple(
        namedtuple('EvaluateTuple',
                   ('returns_mean', 'returns_std', 'sharp', 'turnover',
                    'maxdd', 'returns_mdd', 'win_rate', 'ic', 'ir', 'fitness',
                    'category', 'count', 'ret2tv', 'count_series',
                    'returns_series', 'ic_series', 'turnover_series'))):

    __slots__ = ()

    def __repr__(self):
        return (f"\n--- {self.category} ---"
                f"\nreturns_mean:{self.returns_mean:.6f}"
                f"\nreturns_std:{self.returns_std:.6f}"
                f"\nsharp:{self.sharp:.4f}"
                f"\nturnover:{self.turnover:.4f}"
                f"\nmaxdd:{self.maxdd:.4f}"
                f"\nreturns_mdd:{self.returns_mdd:.4f}"
                f"\nwin_rate:{self.win_rate:.4f}"
                f"\nic:{self.ic:.4f}"
                f"\nir:{self.ir:.4f}"
                f"\nret2tv:{self.ret2tv:.4f}"
                f"\nfitness:{self.fitness:.4f}"
                f"\ncount:{self.count:.1f}")


class MetricsTuple(
        namedtuple('MetricsTuple',
                   ('long_evaluate', 'short_evaluate', 'both_evaluate',
                    'topn_evaluate', 'hold', 'freq', 'direction', 'bias',
                    'category', 'top_n'))):
    __slots__ = ()

    def __repr__(self):
        return (f"long_evaluate:{self.long_evaluate}\n"
                f"short_evaluate:{self.short_evaluate}\n"
                f"both_evaluate:{self.both_evaluate}\n"
                f"topn_evaluate:{self.topn_evaluate}\n"
                f"hold:{self.hold}, freq:{self.freq}, "
                f"direction:{self.direction}, bias:{self.bias:.2f}, "
                f"category:{self.category}, top_n:{self.top_n}")


def valid_check(func):
    """检测度量的输入是否正常"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.valid:
            return func(self, *args, **kwargs)
        else:
            print('[WARN] Metrics input is invalid or zero order gen!')
            return None

    return wrapper


class Metrics(object):
    """
    Cython 加速版因子截面评估器。
    
    用法与 csm001.Metrics 完全相同，核心计算由 Cython Booster 执行。
    """

    @classmethod
    def general(cls,
                returns,
                factors,
                hold=1,
                skip=0,
                top_n=20,
                dummy=None,
                direction=None,
                category=EXCESS,
                freq=DALIY_PER_YEAR,
                fee=0.0,
                show_log=True,
                is_series=False,
                topn_weight_method='factor'):
        """工厂方法：一行调用完成全量评估。"""
        metrics = cls(returns=returns,
                      factors=factors,
                      hold=hold,
                      freq=freq,
                      direction=direction,
                      category=category,
                      skip=skip,
                      top_n=top_n,
                      dummy=dummy,
                      fee=fee,
                      show_log=show_log,
                      is_series=is_series,
                      topn_weight_method=topn_weight_method)
        return metrics.fit_metrics()

    def __init__(self,
                 returns,
                 factors,
                 hold=1,
                 direction=None,
                 category=EXCESS,
                 freq=DALIY_PER_YEAR,
                 dummy=None,
                 skip=0,
                 top_n=20,
                 fee=0.0,
                 show_log=True,
                 is_series=False,
                 topn_weight_method='factor'):
        self.valid = False
        self.category = category
        self.skip = skip
        self.top_n = top_n
        self.fee = fee
        self.freq = freq
        self.hold = hold
        self.show_log = show_log
        self.is_series = is_series
        self.direction = direction
        self.topn_weight_method = topn_weight_method

        # 保存 pandas 引用 (用于输出)
        self._returns_index = returns.index
        self._returns_columns = returns.columns

        # 创建 Cython Booster
        self.booster = Booster(hold, skip, top_n, category)

        # 转 numpy 并预处理
        dummy_vals = dummy.values if dummy is not None else None
        self.ereturns = self.booster.yields(returns.values.copy(), dummy_vals,
                                            skip, category)
        self.score_vals = self.booster.score(
            factors.values.copy(),
            dummy_vals)

        if self.ereturns is not None and self.score_vals is not None:
            self.valid = True

    def _make_evaluate_tuple(self, indicator, ic_arr, ic_mean, ic_std,
                             weight, category):
        """将 booster 输出转换为 EvaluateTuple"""
        (rets_sum, rets_mean, rets_std, sharp, turnover, maxdd,
         ret2mdd, ret2tv, win_rate, fitness, turnover_series,
         count_series) = indicator

        ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
        count = np.mean(count_series)

        # 可选时序输出
        if self.is_series:
            returns_series = pd.Series(rets_sum,
                                       index=self._returns_index,
                                       name='returns')
            ic_series = pd.Series(ic_arr,
                                  index=self._returns_index,
                                  name='ic')
            tv_series = pd.Series(turnover_series,
                                  index=self._returns_index[1:],
                                  name='turnover')
            cnt_series = pd.Series(count_series,
                                   index=self._returns_index,
                                   name='count')
        else:
            returns_series = None
            ic_series = None
            tv_series = None
            cnt_series = None

        return EvaluateTuple(
            returns_mean=rets_mean,
            returns_std=rets_std,
            sharp=sharp,
            turnover=turnover,
            maxdd=maxdd,
            returns_mdd=ret2mdd,
            win_rate=win_rate,
            ic=ic_mean,
            ir=ir,
            ret2tv=ret2tv,
            fitness=fitness,
            count=count,
            category=category,
            count_series=cnt_series,
            returns_series=returns_series,
            ic_series=ic_series,
            turnover_series=tv_series)

    def _apply_hold_smoothing(self, weight):
        """apply rolling hold smoothing on numpy weight"""
        if self.hold > 1:
            weight_df = pd.DataFrame(weight,
                                     index=self._returns_index,
                                     columns=self._returns_columns)
            weight_df = weight_df.rolling(self.hold,
                                          min_periods=1).sum() / self.hold
            return weight_df.values
        return weight

    def _apply_fee(self, rets_sum, weight):
        """扣除交易费用"""
        if self.fee > 0:
            tv = np.nansum(np.abs(weight[1:] - weight[:-1]), axis=1) * 0.5
            tv_full = np.concatenate([[0.0], tv])
            rets_sum = rets_sum - self.fee * tv_full
        return rets_sum

    @valid_check
    def fit_metrics(self):
        """执行全量评估"""
        score = self.score_vals

        # ---------- Long / Short / Both ----------
        right = self.booster.create_weight(score, is_pos=True)
        left = self.booster.create_weight(score, is_pos=False)

        right = self._apply_hold_smoothing(right)
        left = self._apply_hold_smoothing(left)

        long_weight, short_weight, both_weight, direction_val = \
            self.booster.direction(right, left, self.ereturns)

        if self.direction is not None:
            direction_val = self.direction

        # Evaluate each side
        long_ind = self.booster.evaluate(long_weight, self.ereturns,
                                         self.hold, self.freq)
        long_ic, long_ic_mean, long_ic_std = self.booster.correlation(
            long_weight, self.ereturns, 'long')
        long_evaluate = self._make_evaluate_tuple(
            long_ind, long_ic, long_ic_mean, long_ic_std,
            long_weight, OLNY_LONG)

        short_ind = self.booster.evaluate(short_weight, self.ereturns,
                                          self.hold, self.freq)
        short_ic, short_ic_mean, short_ic_std = self.booster.correlation(
            short_weight, self.ereturns, 'short')
        short_evaluate = self._make_evaluate_tuple(
            short_ind, short_ic, short_ic_mean, short_ic_std,
            short_weight, OLNY_SHORT)

        both_ind = self.booster.evaluate(both_weight, self.ereturns,
                                         self.hold, self.freq)
        both_ic, both_ic_mean, both_ic_std = self.booster.correlation(
            both_weight, self.ereturns, 'both')
        both_evaluate = self._make_evaluate_tuple(
            both_ind, both_ic, both_ic_mean, both_ic_std,
            both_weight, BOTH_SIDE)

        # ---------- TopN ----------
        topn_weight = self.booster.create_topn_weight(
            score, self.top_n, self.topn_weight_method)
        topn_weight = self._apply_hold_smoothing(topn_weight)

        # re-normalize after smoothing
        if self.hold > 1:
            sums = np.nansum(topn_weight, axis=1, keepdims=True)
            topn_weight = np.divide(topn_weight, sums,
                                    where=sums > 0, out=topn_weight)

        topn_ind = self.booster.evaluate(topn_weight, self.ereturns,
                                         self.hold, self.freq)
        topn_ic, topn_ic_mean, topn_ic_std = self.booster.correlation(
            topn_weight, self.ereturns, 'long')
        topn_evaluate = self._make_evaluate_tuple(
            topn_ind, topn_ic, topn_ic_mean, topn_ic_std,
            topn_weight, TOP_N)

        self.direction = direction_val

        if self.show_log:
            print(f"\n{'=' * 50}")
            print(f"  ArbMetrics Report [Cython]  (top_n={self.top_n}, "
                  f"hold={self.hold}, freq={self.freq})")
            print(f"{'=' * 50}")
            print(long_evaluate)
            print(short_evaluate)
            print(both_evaluate)
            print(topn_evaluate)
            print(f"\ndirection={self.direction}")
            print(f"{'=' * 50}")

        return MetricsTuple(
            long_evaluate=long_evaluate,
            short_evaluate=short_evaluate,
            both_evaluate=both_evaluate,
            topn_evaluate=topn_evaluate,
            freq=self.freq,
            hold=self.hold,
            bias=(long_evaluate.count /
                  short_evaluate.count
                  if short_evaluate.count != 0 else 0),
            category=self.category,
            direction=self.direction,
            top_n=self.top_n)
