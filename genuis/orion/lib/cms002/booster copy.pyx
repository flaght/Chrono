import copy
import numpy as np
cimport numpy as np
from scipy.stats import norm


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
            ret_mkt_fnd = np.nanmean(shifted_returns, axis=1)
            for i in range(shifted_returns.shape[0]):
                shifted_returns[i, :] -= ret_mkt_fnd[i]
        return shifted_returns

    cpdef score(self, factors, dummy):
        """因子截面正态化：rank → probit"""
        cdef invar = factors * dummy if isinstance(dummy, np.ndarray) else factors
        cdef ranked = np.empty_like(invar, dtype=np.float64)
        cdef valid_mask
        cdef valid_values
        cdef sorted_indices
        cdef ranks
        cdef count

        ranked.fill(np.nan)
        for row in range(invar.shape[0]):
            valid_mask = ~np.isnan(invar[row, :])
            valid_values = invar[row, valid_mask]
            if valid_values.size == 0:
                continue
            sorted_indices = np.argsort(valid_values)
            ranks = np.empty_like(sorted_indices, dtype=np.float64)
            ranks[sorted_indices] = np.arange(1, len(valid_values) + 1)
            ranked[row, valid_mask] = ranks
        count = np.sum(~np.isnan(ranked), axis=1)
        ranked = (ranked - 3. / 8.) / (count[:, None] + 1. / 4.)
        return norm.ppf(ranked)

    cpdef create_weight(self, factors, is_pos=True):
        """创建多头或空头权重"""
        cdef weight = copy.deepcopy(factors)
        cdef sums

        if is_pos:
            weight[weight <= 0] = np.nan
        else:
            weight[weight >= 0] = np.nan
        sums = np.nansum(weight, axis=1, keepdims=True)
        weight = np.divide(weight, sums, where=sums != 0)
        return weight

    cpdef create_topn_weight(self, factors, int top_n, str weight_method):
        """创建 TopN 权重"""
        cdef int T = factors.shape[0]
        cdef int N = factors.shape[1]
        cdef topn_weight = np.full((T, N), np.nan, dtype=np.float64)
        cdef row_vals
        cdef valid_mask
        cdef valid_indices
        cdef sorted_idx
        cdef topn_idx
        cdef vals
        cdef double s

        for i in range(T):
            row_vals = factors[i, :]
            valid_mask = ~np.isnan(row_vals)
            valid_indices = np.where(valid_mask)[0]
            if valid_indices.size == 0:
                continue
            sorted_idx = valid_indices[np.argsort(-row_vals[valid_indices])]
            topn_idx = sorted_idx[:min(top_n, len(sorted_idx))]

            if weight_method == 'equal':
                topn_weight[i, topn_idx] = 1.0 / len(topn_idx)
            elif weight_method == 'factor':
                vals = np.clip(row_vals[topn_idx], 0, None)
                s = np.sum(vals)
                if s > 0:
                    topn_weight[i, topn_idx] = vals / s
                else:
                    topn_weight[i, topn_idx] = 1.0 / len(topn_idx)
            elif weight_method == 'sqrt':
                vals = np.sqrt(np.clip(row_vals[topn_idx], 0, None))
                s = np.sum(vals)
                if s > 0:
                    topn_weight[i, topn_idx] = vals / s
                else:
                    topn_weight[i, topn_idx] = 1.0 / len(topn_idx)

        return topn_weight

    cpdef direction(self, right_weight, left_weight, returns, fill_value=0.0):
        """自动判断因子方向"""
        cdef right_returns = np.nansum(returns * right_weight, axis=1)
        cdef left_returns = np.nansum(returns * left_weight, axis=1)
        cdef diff = np.mean(right_returns) - np.mean(left_returns)
        cdef long_weight
        cdef short_weight
        cdef both_weight

        if diff > 0:
            long_weight = right_weight
            short_weight = left_weight
        else:
            long_weight = left_weight
            short_weight = right_weight
        # Fix: 用 sub(fill_value=0) 语义：只把 NaN 当 0 做差，
        # 结果中两边都是 NaN 的位置保持 NaN（避免 count 膨胀）
        both_weight = np.where(
            np.isnan(long_weight) & np.isnan(short_weight),
            np.nan,
            np.nan_to_num(long_weight, nan=0.0) - np.nan_to_num(short_weight, nan=0.0)
        )
        return long_weight, short_weight, both_weight, 1 if diff > 0 else -1

    cpdef correlation(self, weight, returns, str method):
        """
        截面 RankIC (Spearman)：对每个时间步 t，计算 weight[t,:] 与 returns[t,:] 的相关系数。
        与 pandas weight.corrwith(returns, axis=1, method='spearman') 等价。
        """
        cdef int T = weight.shape[0]
        cdef icArray = []
        cdef w_t
        cdef r_t
        cdef mask
        cdef corr

        for i in range(T):
            w_t = weight[i, :]
            r_t = returns[i, :]
            mask = ~np.isnan(w_t) & ~np.isnan(r_t)
            if np.sum(mask) > 1:
                # Spearman = Pearson on ranks
                w_rank = np.argsort(np.argsort(w_t[mask])).astype(np.float64)
                r_rank = np.argsort(np.argsort(r_t[mask])).astype(np.float64)
                corr = np.corrcoef(w_rank, r_rank)[0, 1]
            else:
                corr = np.nan
            icArray.append(corr)
        icArray = np.array(icArray)
        return icArray, np.nanmean(icArray), np.nanstd(icArray)

    cpdef bias(self, right_count, left_count):
        return np.mean(right_count) / np.mean(left_count) if np.mean(left_count) != 0.0 else 0.0

    cpdef evaluate(self, weight, returns, int hold, int freq):
        """核心评估：计算组合绩效全部指标"""
        cdef rets_sum = np.nansum(returns * weight, axis=1)
        cdef int total_periods = np.sum(~np.isnan(rets_sum))
        cdef double years = total_periods / <double>freq if freq > 0 else 1.0
        cdef double total_log_ret = np.nansum(rets_sum)
        cdef double rets_mean = total_log_ret / years if years > 0 else 0.0
        cdef double rets_std = np.nanstd(rets_sum) * np.sqrt(freq)
        cdef double sharp = rets_mean / rets_std if rets_std > 1e-10 else 0.0
        # Fix: NaN→0 before diff，使入场/出场的换手被正确计入，与 pandas fill_value=0 一致
        cdef weight0 = np.nan_to_num(weight, nan=0.0)
        cdef turnover_series = np.sum(np.abs(weight0[1:] - weight0[:-1]), axis=1) * 0.5
        cdef double turnover = np.mean(turnover_series)
        cdef pnl = np.nancumsum(rets_sum)
        cdef double maxdd = np.nanmax(np.maximum.accumulate(pnl) - pnl)
        cdef double ret2mdd = rets_mean / maxdd if maxdd > 1e-10 else 0.0
        cdef double ret2tv = rets_mean / turnover if turnover > 1e-10 else 0.0
        cdef double win_rate = np.sum(rets_sum > 0) / np.sum(~np.isnan(rets_sum)) if np.sum(~np.isnan(rets_sum)) != 0 else 0.0
        cdef double fitness = sharp * np.sqrt(np.abs(rets_mean) / turnover) if turnover > 1e-10 else 0.0
        cdef count_series = np.count_nonzero(~np.isnan(weight), axis=1)

        return (rets_sum, rets_mean, rets_std, sharp, turnover, maxdd,
                ret2mdd, ret2tv, win_rate, fitness, turnover_series,
                count_series)
