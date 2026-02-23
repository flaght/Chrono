"""
核心指标：
        1. RankIC   — 截面排序预测力
        2. TopN     — 头部赚钱能力
        3. Turnover — 可执行性
        4. Sharpe   — 净收益质量
"""
import copy, pdb
import functools
import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import namedtuple

DALIY_PER_YEAR = 252
WEEKLY_PER_YEAR = 52
MONTHLY_PER_YEAR = 12
QUARTERLY_PER_YEAR = 4
YEARLY_PER_YEAR = 1
HOURLY_PER_YEAR = 365 * 24  # 8760, 加密货币全年无休

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
            print('[WARN] ArbMetrics input is invalid or zero order gen!')
            return None

    return wrapper


class Metrics(object):
    """
    因子截面评估器。
    Parameters
    ----------
    returns : pd.DataFrame (T x N)
        持有期收益率。**必须是 log return (对数收益率)**。
        例如：如果价格从 100 涨到 101，log return = ln(101/100) ≈ 0.00995。
        注意：不要输入 simple return (101/100 - 1 = 0.01)。
    factors : pd.DataFrame (T x N)
        因子值。值越大代表越看好。
    hold : int, default 1
        持仓期数。hold=1 表示每期调仓。
    skip : int, default 0
        信号延迟期。skip=1 表示 t 时刻信号在 t+1 才执行。
    top_n : int, default 20
        TopN 组合选取的标的数量。
    dummy : pd.DataFrame (T x N), optional
        可交易掩码。1=可交易, 0/NaN=不可交易。
    direction : int, optional
        强制因子方向。POSITIVE=1 (因子大=做多), NEGATIVE=-1。
        None=自动判断。
    category : int, default EXCESS
        EXCESS=1: 使用截面超额收益; ABSOLUTE=-1: 使用绝对收益。
    freq : int, default DALIY_PER_YEAR
        年化因子。日频=252, 小时频=8760。
    fee : float, default 0.0
        单次换手交易成本。
    show_log : bool, default True
        是否打印详细日志。
    is_series : bool, default False
        是否在结果中返回时间序列 (returns_series, ic_series 等)。
    method : str, default 'max'
        截面分数化方法。'std' 或 'max'。
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
                 method='max',
                 topn_weight_method='factor'):
        self.valid = False
        self.category = category
        self.dummy = dummy
        self.skip = skip
        self.top_n = top_n
        self.fee = fee
        self.returns = self.yields(returns)
        self.factors = self.score(factors, method)
        self.freq = freq
        self.hold = hold
        self.show_log = show_log
        self.is_series = is_series
        self.direction = direction
        self.topn_weight_method = topn_weight_method
        if self.returns is not None and self.factors is not None:
            self.valid = True

    # fixed 加上超额收益和绝对收益
    def yields(self, returns):
        """
        收益率预处理：掩码 → 延迟 → 截面中性化。
        
        注意：输入必须是 log return，此方法不做 exp 转换。
        """
        returns = returns * self.dummy if self.dummy is not None else returns
        returns = returns.shift(-self.skip)
        if self.category == EXCESS:
            ret_mkt_fnd = returns.mean(axis=1)
            return returns.sub(ret_mkt_fnd, axis='rows')
        else:
            return returns  # 绝对收益

    def score(self, factors, method='std'):
        """因子截面分数化：rank → 归一化"""
        factors = factors * self.dummy.shift(
            -self.skip) if self.dummy is not None else factors
        if method == 'std':
            rank = factors.rank(axis=1, method='dense')
            score = (rank - 0.5).div(rank.max(axis=1), axis='rows') - 0.5
            return score.pow(3)
        else:
            rank = factors.rank(axis=1, method='max')
            count = rank.count(axis=1)
            rank = (rank - 3. / 8.).div(count + 1. / 4., axis='rows')
            score = pd.DataFrame(norm.ppf(rank),
                                 index=rank.index,
                                 columns=rank.columns)
            return score

    def create_weight(self):
        """基于因子分数正负拆分为多头/空头/多空联合权重。"""
        right_weight = copy.deepcopy(self.factors)
        right_weight[right_weight <= 0] = np.nan
        right_weight = right_weight.div(right_weight.sum(axis=1, min_count=1),
                                        axis='rows')
        right_weight = right_weight.rolling(self.hold,
                                            min_periods=1).sum() / self.hold

        left_weight = copy.deepcopy(self.factors)
        left_weight[left_weight >= 0] = np.nan
        left_weight = left_weight.div(left_weight.sum(axis=1, min_count=1),
                                      axis='rows')
        left_weight = left_weight.rolling(
            self.hold,
            min_periods=1).sum() / self.hold  # fixed 为什么要和持仓周期滚动 做平滑？

        both_weight = right_weight.sub(left_weight, fill_value=0)

        ret_diff = ((self.returns * right_weight).sum(axis=1).mean() -
                    (self.returns * left_weight).sum(axis=1).mean())

        if ret_diff > 0 and self.direction is None:
            self.direction = POSITIVE
        elif ret_diff < 0 and self.direction is None:
            self.direction = NEGATIVE

        if self.direction == POSITIVE:
            long_weight = copy.deepcopy(right_weight)
            short_weight = copy.deepcopy(left_weight)
            both_weight = copy.deepcopy(both_weight)
        elif self.direction == NEGATIVE:
            long_weight = copy.deepcopy(left_weight)
            short_weight = copy.deepcopy(right_weight)
            both_weight = copy.deepcopy(-both_weight)

        return long_weight, short_weight, both_weight

    ## fixed 新增等权和加权区分
    def create_topn_weight(self, weight_method='equal'):
        """
        基于因子分数选取每期 TopN 个标的，等权持有。
        逻辑：
            1. 对每个时间点，按因子值降序排名
            2. 选取排名 <= top_n 的标的
            3. 权重 
            4. 如果 hold > 1，滚动平均权重平滑调仓

        weight_method : str, default 'equal'
        权重分配方式：
        - 'equal'  : 等权 1/N
        - 'factor' : 按因子值比例加权
        - 'sqrt'   : 按因子值开方后加权 (介于等权和因子加权之间)

        Returns
        -------
        pd.DataFrame : TopN 等权权重矩阵 (T x N)
        """
        N = self.top_n
        factors = self.factors

        # 1. 选出 TopN
        rank_desc = factors.rank(axis=1,
                                 ascending=False,
                                 method='first',
                                 na_option='bottom')
        topn_mask = (rank_desc <= N).astype(float)
        topn_mask = topn_mask.replace(0, np.nan)

        if weight_method == 'equal':  # 等权: 1/N
            topn_weight = topn_mask.div(topn_mask.count(axis=1), axis='rows')
        elif weight_method == 'factor':
            topn_factors = factors * topn_mask  # 只保留 TopN 的因子值
            # 归一化到 [0, 1]，避免负值
            topn_factors_pos = topn_factors.clip(lower=0)
            topn_weight = topn_factors_pos.div(topn_factors_pos.sum(
                axis=1, min_count=1),
                                               axis='rows')
        elif weight_method == 'sqrt':
            # 开方加权 (降低头部集中度)
            topn_factors = factors * topn_mask
            topn_factors_pos = topn_factors.clip(lower=0)
            topn_factors_sqrt = np.sqrt(topn_factors_pos)
            topn_weight = topn_factors_sqrt.div(topn_factors_sqrt.sum(
                axis=1, min_count=1),
                                                axis='rows')

        # 持仓期平滑
        if self.hold > 1:
            topn_weight = topn_weight.rolling(self.hold,
                                              min_periods=1).sum() / self.hold
            # 重新归一化
            topn_weight = topn_weight.div(topn_weight.sum(axis=1, min_count=1),
                                          axis='rows')

        return topn_weight

    def evaluate(self,
                 returns,
                 weight,
                 freq=DALIY_PER_YEAR,
                 category=OLNY_LONG):
        """
        核心评估方法 给定权重矩阵，计算组合绩效的全部指标。
        指标清单：
            - returns_mean : 年化收益
            - returns_std  : 年化波动
            - sharp        : 夏普比率          ← Sharpe
            - turnover     : 平均换手率        ← Turnover
            - maxdd        : 最大回撤
            - returns_mdd  : 收益回撤比
            - win_rate     : 胜率
            - ic           : 截面 RankIC 均值  ← RankIC
            - ir           : IC 信息比率
            - fitness      : 综合适应度
            - calma       : 收益/换手比
            - count        : 平均持仓数量
        """
        # returns 已经是 log return，加权求和即可
        rets_sum = (returns * weight).sum(axis=1, min_count=1)

        tv_series = abs(weight.sub(weight.shift(1), fill_value=0)).sum(
            axis=1, min_count=1) * 0.5

        if self.fee > 0:
            net_ret = rets_sum - self.fee * tv_series.fillna(0)
        else:
            net_ret = rets_sum

        # ---- 年化统计 ----
        # ---- 年化统计 (针对 log return) ----
        # log return 的年化：几何平均
        # 总收益 = sum(log returns) = log(累计净值)
        # 年化收益 = 总收益 / 年数
        total_periods = len(net_ret.dropna())
        if total_periods > 0:
            years = total_periods / freq
            total_log_ret = net_ret.sum()
            rets_mean = total_log_ret / years if years > 0 else 0
        else:
            rets_mean = 0

        rets_std = net_ret.std() * np.sqrt(freq)
        sharp = rets_mean / rets_std if rets_std > 1e-10 else 0

        tv = tv_series.mean()
        count = weight.count(axis=1).mean()

        pnl = net_ret.cumsum()
        maxdd = (pnl.expanding().max() - pnl).max()
        ret2mdd = rets_mean / maxdd if maxdd > 1e-10 else 0
        ret2tv = rets_mean / tv if tv > 1e-10 else 0

        win_rate = (net_ret[net_ret > 0].count() /
                    net_ret[~net_ret.isna()].count()
                    if net_ret[~net_ret.isna()].count() > 0 else 0)

        # ---- 截面 RankIC (Spearman) ----
        ic_series = weight.corrwith(returns, axis=1, method='spearman')
        ic = ic_series.mean()
        ir = ic / ic_series.std() if ic_series.std() > 1e-10 else 0

        # ---- 综合适应度 ----
        fitness = (sharp * np.sqrt(abs(rets_mean) / tv) if tv > 1e-10 else 0)

        # ---- 可选时序输出 ----
        returns_series = net_ret if self.is_series else None
        ic_out = ic_series if self.is_series else None
        turnover_series = tv_series if self.is_series else None
        count_series = weight.count(axis=1) if self.is_series else None

        return EvaluateTuple(returns_mean=rets_mean,
                             returns_std=rets_std,
                             sharp=sharp,
                             turnover=tv,
                             maxdd=maxdd,
                             returns_mdd=ret2mdd,
                             win_rate=win_rate,
                             ic=ic,
                             ir=ir,
                             ret2tv=ret2tv,
                             fitness=fitness,
                             count=count,
                             category=category,
                             count_series=count_series,
                             returns_series=returns_series,
                             ic_series=ic_out,
                             turnover_series=turnover_series)

    @valid_check
    def fit_metrics(self):
        """
        执行全量评估。

        输出 4 个 EvaluateTuple:
            - long_evaluate  : 分数>0 加权多头组合
            - short_evaluate : 分数<0 加权空头组合
            - both_evaluate  : 多空联合组合
            - topn_evaluate  : TopN 等权组合 (新增)
        """
        # ---------- Long / Short / Both (原 metrics.py 逻辑) ----------
        long_weight, short_weight, both_weight = self.create_weight()

        long_evaluate = self.evaluate(returns=self.returns,
                                      weight=long_weight,
                                      freq=self.freq,
                                      category=OLNY_LONG)

        short_evaluate = self.evaluate(returns=self.returns,
                                       weight=short_weight,
                                       freq=self.freq,
                                       category=OLNY_SHORT)

        both_evaluate = self.evaluate(returns=self.returns,
                                      weight=both_weight,
                                      freq=self.freq,
                                      category=BOTH_SIDE)

        # ---------- TopN (新增) ----------
        topn_weight = self.create_topn_weight(self.topn_weight_method)

        topn_evaluate = self.evaluate(returns=self.returns,
                                      weight=topn_weight,
                                      freq=self.freq,
                                      category=TOP_N)

        if self.show_log:
            print(f"\n{'=' * 50}")
            print(f"  ArbMetrics Report  (top_n={self.top_n}, "
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
                  short_evaluate.count if short_evaluate.count != 0 else 0),
            category=self.category,
            direction=self.direction,
            top_n=self.top_n)