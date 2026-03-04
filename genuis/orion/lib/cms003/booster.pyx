import copy
import numpy as np
cimport numpy as np
from scipy.stats import norm
import pandas as pd
import warnings

cdef class Booster(object):
    cdef public int _hold
    cdef public int _skip
    cdef public int _top_n
    cdef public int _category

    def __init__(self, hold, skip, top_n, category):
        self._hold = hold
        self._skip = skip
        self._top_n = top_n
        self._category = category

    cpdef yields(self, returns, dummy, int skip, int category):
        """收益率预处理：掩码 → 延迟 → 截面中性化"""
        cdef rets = returns * dummy if isinstance(dummy, np.ndarray) else returns
        cdef shifted_returns = rets.copy()
        cdef ret_mkt_fnd

        # skip 偏移
        if skip > 0:
            shifted_returns[:-skip] = shifted_returns[skip:]
            shifted_returns[-skip:] = np.nan

        if category == 1:  # EXCESS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # 使用 keepdims=True 避免 for 循环
                ret_mkt_fnd = np.nanmean(shifted_returns, axis=1, keepdims=True)
            shifted_returns -= ret_mkt_fnd
        return shifted_returns

    cpdef score(self, factors, dummy):
        """因子截面正态化：rank → probit (纯 Numpy 向量化版本)"""
        cdef invar = factors * dummy if isinstance(dummy, np.ndarray) else factors
        
        # 记录完全无效的 NaN掩码，NaN将作为 inf 移至 Argsort 的最后面
        cdef valid_mask = ~np.isnan(invar)
        cdef mask_factors = np.where(valid_mask, invar, np.inf)
        
        # Numpy Argsort 取 Rank，等价于 DataFrame.rank(method='first', ascending=True)
        cdef sort_idx = np.argsort(mask_factors, axis=1)
        cdef ranks = np.empty_like(sort_idx, dtype=np.float64)
        cdef row_idx = np.arange(invar.shape[1])
        cdef broadcast_row = np.broadcast_to(row_idx, invar.shape)
        
        np.put_along_axis(ranks, sort_idx, broadcast_row, axis=1)
        ranks = ranks + 1.0  # pandas rank starts from 1
        
        cdef ranked = np.where(valid_mask, ranks, np.nan)
        cdef count = valid_mask.sum(axis=1, keepdims=True)
        
        ranked = (ranked - 3. / 8.) / (count + 1. / 4.)
        return norm.ppf(ranked)

    cpdef percent_rank(self, factors):
        """计算截面分位数 (纯 Numpy 向量化版本替代 pd.DataFrame.rank(pct=True))"""
        cdef valid_mask = ~np.isnan(factors)
        cdef mask_factors = np.where(valid_mask, factors, np.inf)
        
        cdef sort_idx = np.argsort(mask_factors, axis=1)
        cdef ranks = np.empty_like(sort_idx, dtype=np.float64)
        cdef int N = factors.shape[1]
        cdef row_idx = np.arange(N)
        cdef broadcast_row = np.broadcast_to(row_idx, factors.shape)
        
        np.put_along_axis(ranks, sort_idx, broadcast_row, axis=1)
        ranks = ranks + 1.0  # Rank starts from 1
        
        cdef counts = valid_mask.sum(axis=1, keepdims=True)
        cdef pct_ranks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pct_ranks = np.divide(ranks, counts, out=np.full_like(ranks, np.nan), where=(counts > 0))
            
        return np.where(valid_mask, pct_ranks, np.nan)

    cpdef create_weight(self, factors, is_pos=True):
        """创建多头或空头权重"""
        cdef weight = copy.deepcopy(factors)
        cdef sums

        if is_pos:
            weight[weight <= 0] = np.nan
        else:
            weight[weight >= 0] = np.nan
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sums = np.nansum(weight, axis=1, keepdims=True)
            
        weight = np.divide(weight, sums, out=weight, where=(sums != 0))
        return weight

    cpdef create_topn_weight(self, factors, int top_n, str weight_method):
        """创建 TopN 权重 (纯 Numpy 向量化版本)"""
        cdef int T = factors.shape[0]
        cdef int N = factors.shape[1]
        
        cdef valid_mask = ~np.isnan(factors)
        # 取负因子，将 NaN 设置为 inf 排到最后面，获取最大值的 Argsort
        cdef neg_factors = np.where(valid_mask, -factors, np.inf)
        cdef sort_idx = np.argsort(neg_factors, axis=1)
        cdef ranks = np.empty_like(sort_idx, dtype=np.float64)
        cdef row_idx = np.arange(N)
        cdef broadcast_row = np.broadcast_to(row_idx, (T, N))
        
        np.put_along_axis(ranks, sort_idx, broadcast_row, axis=1)
        ranks = ranks + 1.0
        
        # 屏蔽未进入 Top_n 的元素和本身的 NaN
        cdef mask = (ranks <= top_n) & valid_mask
        cdef counts = mask.sum(axis=1, keepdims=True)
        cdef vals, sums, wt, equal_wt, factor_wt, sqrt_wt
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            equal_wt = np.divide(1.0, counts, out=np.zeros_like(counts, dtype=np.float64), where=(counts > 0))

        if weight_method == 'equal':
            wt = np.broadcast_to(equal_wt, (T, N))
        elif weight_method == 'factor':
            vals = np.where(mask, factors, 0.0)
            vals = np.clip(vals, 0, None)
            sums = np.sum(vals, axis=1, keepdims=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                factor_wt = np.divide(vals, sums, out=np.zeros_like(vals), where=(sums > 0))
            wt = np.where(sums > 0, factor_wt, equal_wt)
        elif weight_method == 'sqrt':
            vals = np.where(mask, factors, 0.0)
            vals = np.sqrt(np.clip(vals, 0, None))
            sums = np.sum(vals, axis=1, keepdims=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sqrt_wt = np.divide(vals, sums, out=np.zeros_like(vals), where=(sums > 0))
            wt = np.where(sums > 0, sqrt_wt, equal_wt)
            
        return np.where(mask, wt, np.nan)

    cpdef direction(self, right_weight, left_weight, returns, fill_value=0.0):
        """自动判断因子方向"""
        cdef right_returns, left_returns, diff
        cdef long_weight, short_weight, both_weight
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            right_returns = np.nansum(returns * right_weight, axis=1)
            left_returns = np.nansum(returns * left_weight, axis=1)
            diff = np.nanmean(right_returns) - np.nanmean(left_returns)

        if diff > 0:
            long_weight = right_weight
            short_weight = left_weight
        else:
            long_weight = left_weight
            short_weight = right_weight

        both_weight = np.where(
            np.isnan(long_weight) & np.isnan(short_weight),
            np.nan,
            np.nan_to_num(long_weight, nan=0.0) - np.nan_to_num(short_weight, nan=0.0)
        )
        return long_weight, short_weight, both_weight, 1 if diff > 0 else -1

    cpdef correlation(self, weight, returns, str method):
        """截面 RankIC (Spearman) - Numpy 纯向量化相关系数"""
        cdef valid_pair = ~np.isnan(weight) & ~np.isnan(returns)
        cdef int T = weight.shape[0]
        cdef int N = weight.shape[1]
        
        # 仅对存在配对的位置取 rank，其它为 np.inf 被排至末尾
        cdef W = np.where(valid_pair, weight, np.inf)
        cdef R = np.where(valid_pair, returns, np.inf)

        # 第一次 Argsort: np.inf 全被排到后面，其余的数值正常拿到名次索引
        # 第二次利用 np.put_along_axis 拿出名次
        cdef W_sort = np.argsort(W, axis=1)
        cdef R_sort = np.argsort(R, axis=1)

        cdef W_rank = np.empty_like(W_sort, dtype=np.float64)
        cdef R_rank = np.empty_like(R_sort, dtype=np.float64)

        cdef row_idx = np.arange(N)
        cdef broadcast_row = np.broadcast_to(row_idx, (T, N))
        
        np.put_along_axis(W_rank, W_sort, broadcast_row, axis=1)
        np.put_along_axis(R_rank, R_sort, broadcast_row, axis=1)

        # 恢复由于有效部分取得的 Rank 加上了 1，无效位置继续掩盖为 NaN
        W_rank = np.where(valid_pair, W_rank + 1.0, np.nan)
        R_rank = np.where(valid_pair, R_rank + 1.0, np.nan)

        # 截面求相关系数 Pearson (on rank = Spearman)
        # corr(x, y) = cov(x,y) / (std(x)*std(y)) -> sum( (x-u_x)*(y-u_y) ) / sqrt( sum(x-u_x)^2 * sum(y-u_y)^2 )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            W_mean = np.nanmean(W_rank, axis=1, keepdims=True)
            R_mean = np.nanmean(R_rank, axis=1, keepdims=True)

            W_diff = W_rank - W_mean
            R_diff = R_rank - R_mean

            cov = np.nansum(W_diff * R_diff, axis=1)
            W_var = np.nansum(W_diff**2, axis=1)
            R_var = np.nansum(R_diff**2, axis=1)
            
            # 分母可能是 0 如果某个截面上无有效因子或无方差
            icArray = np.divide(cov, np.sqrt(W_var * R_var), out=np.full(T, np.nan), where=(W_var * R_var > 0))

        return icArray, np.nanmean(icArray), np.nanstd(icArray)

    cpdef bias(self, right_count, left_count):
        return np.mean(right_count) / np.mean(left_count) if np.mean(left_count) != 0.0 else 0.0

    cpdef evaluate(self, weight, returns, int hold, int freq, object date_groups=None, int num_groups=0):
        """核心评估：计算组合绩效全部指标"""
        # 所有的 cdef 声明必须前置在方法的最顶部
        cdef rets_sum, total_periods, years, total_log_ret, rets_mean, rets_std, sharp
        cdef weight0, turnover_series, turnover, pnl, maxdd, ret2mdd
        cdef calmar, valid_rets_count, win_rate, fitness, count_series
        cdef valid_mask, daily_rets, daily_valid, valid_daily_rets
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rets_sum = np.nansum(returns * weight, axis=1)
            
            if date_groups is not None and num_groups > 0:
                # 按天归集降频
                daily_rets = np.bincount(date_groups, weights=np.nan_to_num(rets_sum, nan=0.0))
                valid_mask = ~np.isnan(rets_sum)
                daily_valid = np.bincount(date_groups, weights=valid_mask)
                
                # 仅考虑有交易/信号的有效天数
                valid_daily_rets = daily_rets[daily_valid > 0]
                
                total_periods = len(valid_daily_rets)
                years = total_periods / float(freq) if freq > 0 else 1.0 # 此时 freq 代表 annual_days (如250)
                
                total_log_ret = np.nansum(valid_daily_rets)
                rets_mean = total_log_ret / years if years > 0 else 0.0
                rets_std = np.nanstd(valid_daily_rets) * np.sqrt(freq)
                sharp = rets_mean / rets_std if rets_std > 1e-10 else 0.0
            else:
                total_periods = np.sum(~np.isnan(rets_sum))
                years = total_periods / float(freq) if freq > 0 else 1.0
                
                total_log_ret = np.nansum(rets_sum)
                rets_mean = total_log_ret / years if years > 0 else 0.0
                rets_std = np.nanstd(rets_sum) * np.sqrt(freq)
                sharp = rets_mean / rets_std if rets_std > 1e-10 else 0.0
            
            weight0 = np.nan_to_num(weight, nan=0.0)
            turnover_series = np.sum(np.abs(weight0[1:] - weight0[:-1]), axis=1) * 0.5
            turnover = np.mean(turnover_series)
            
            pnl = np.nancumsum(rets_sum)
            maxdd = np.nanmax(np.maximum.accumulate(pnl) - pnl)
            
            ret2mdd = rets_mean / maxdd if maxdd > 1e-10 else 0.0
            calmar = rets_mean / maxdd if maxdd > 1e-10 else 0.0
            
            valid_rets_count = np.sum(~np.isnan(rets_sum))
            win_rate = np.sum(rets_sum > 0) / valid_rets_count if valid_rets_count > 0 else 0.0
            
            fitness = sharp * np.sqrt(np.abs(rets_mean) / turnover) if turnover > 1e-10 else 0.0
            count_series = np.count_nonzero(~np.isnan(weight), axis=1)

        return (rets_sum, rets_mean, rets_std, sharp, turnover, maxdd,
                ret2mdd, calmar, win_rate, fitness, turnover_series,
                count_series)
