### 信号评估
import pdb
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


class Signal(object):

    def __init__(self, signal: pd.Series, future_return: pd.Series, window=5):
        self._signal, self._future_return = signal.align(future_return,
                                                         join='inner')
        self._signal = self._signal.squeeze()
        self._future_return = self._future_return.squeeze()
        self._mask = (self._signal != 0)  # # 只计算非零信号的部分
        self._window = window

    '''
    预测能力指标 - Rank IC分析 离散值
    阈值：
        优秀：|IC| > 0.1 且 p < 0.05
        良好：|IC| > 0.05 且 p < 0.05
        及格：|IC| > 0.02 且 p < 0.1
        不及格：其他
        Rank IC 衡量信号与未来收益的秩相关，>0.05有一定预测力，>0.1为优秀。
    '''

    def rankic1(self):
        mask = self._signal.notna() & self._future_return.notna()
        if mask.sum() < 2:
            return np.nan
        return spearmanr(self._signal[mask], self._future_return[mask])

    '''
        预测能力指标 -  分组收益（group_return）
        阈值（以年化极差为例，假设日频，年化约250倍）：
        优秀：年化极差 > 0.10
        良好：年化极差 > 0.05
        及格：年化极差 > 0.02
        不及格：≤ 0.02
        分组收益极差越大，信号区分度越强。
    '''

    def group_return(self, n_groups):
        # 如果唯一值太少，直接用唯一值分组
        unique_vals = self._signal.dropna().unique()
        if len(unique_vals) <= n_groups or self._signal.nunique() <= n_groups:
            grouped_returns = {}
            for i, val in enumerate(sorted(unique_vals)):
                mask = self._signal == val
                if mask.sum() > 0:
                    group_return = self._future_return[mask].mean()
                    grouped_returns[f'group_{i+1}_val_{val}'] = group_return
            return grouped_returns
        else:
            # 连续型信号，正常qcut
            try:
                signal_quantiles = pd.qcut(self._signal,
                                           n_groups,
                                           labels=False,
                                           duplicates='drop')
            except ValueError:
                # 还是分不了组，退回用唯一值分组
                return self._calculate_grouped_returns(
                    self._signal,
                    self._future_return,
                    n_groups=self._signal.nunique())

            grouped_returns = {}
            for i in range(signal_quantiles.nunique()):
                mask = signal_quantiles == i
                if mask.sum() > 0:
                    group_return = self._future_return[mask].mean()
                    grouped_returns[f'group_{i+1}'] = group_return
            return grouped_returns

    '''
    预测能力指标 -- 信号方向与未来收益方向一致的比例。
    阈值：
        优秀：> 0.60
        良好：> 0.55
        及格：> 0.52
        不及格：≤ 0.52
        50%为随机水平，>55%才有统计意义，>60%为高质量信号。
    '''

    def hit_rate(self):
        mask = self._signal != 0
        hit = ((self._signal[mask] * self._future_return[mask])
               > 0).mean() if mask.sum() > 0 else 0.0
        return hit

    '''
    预测能力指标 - 胜率与盈亏比
    胜率阈值：
        优秀：> 0.65
        良好：> 0.60
        及格：> 0.55
        不及格：≤ 0.55
    盈亏比阈值：
        优秀：> 2.0
        良好：> 1.5
        及格：> 1.2
        不及格：≤ 1.2
    '''

    def win_loss(self):
        mask = self._signal != 0
        win_rate = 0.0
        profit_factor = 0.0
        if mask.sum() > 0:
            correct = (self._signal[mask] * self._future_return[mask]) > 0
            win_rate = correct.mean()
            win_avg = self._future_return[mask][correct].mean() if correct.sum(
            ) > 0 else 0
            loss_avg = abs(self._future_return[mask][~correct].mean()) if (
                ~correct).sum() > 0 else 0
            profit_factor = win_avg / loss_avg if loss_avg != 0 else float(
                'inf')
        return win_rate, profit_factor

    '''
    交易效率指标 - 信号稳定性
    信号稳定性（stability）
    阈值：
        优秀：> 0.8
        良好：> 0.6
        及格：> 0.4
        不及格：≤ 0.4
        信号变化越少越稳定，过于频繁变化会导致高交易成本
    '''

    def stability(self):
        """计算信号稳定性（信号变化频率的倒数）"""
        signal_changes = (self._signal != self._signal.shift(1)).sum()
        total_periods = len(self._signal)
        stability = 1 - (signal_changes / total_periods)
        return stability

    '''
    交易效率指标 - 信号换手率
    阈值（越低越好）：
        优秀：< 0.1
        良好：< 0.3
        及格：< 0.5
        不及格：≥ 0.5
        换手率高会导致高交易成本。
    '''

    def turnover(self):
        """计算信号换手率（信号变化幅度）"""
        signal_changes = self._signal.diff().abs()
        turnover = signal_changes.mean()
        return turnover

    '''
    交易效率指标 - 信号一致性
    阈值：
        优秀：> 0.8
        良好：> 0.6
        及格：> 0.4
        不及格：≤ 0.4
        方向持续性高说明信号更有趋势性。
    '''

    def consistency(self):
        """计算信号一致性（信号方向的持续性）"""
        # 计算信号方向的一致性
        signal_direction = np.sign(self._signal)
        direction_changes = (signal_direction
                             != signal_direction.shift(1)).sum()
        total_periods = len(self._signal)
        consistency = 1 - (direction_changes / total_periods)
        return consistency

    '''
    交易效率指标 - 信号持续性 --离散型
    阈值（最大连续相同信号长度/总长度）：
        优秀：> 0.3
        良好：> 0.2
        及格：> 0.1
        不及格：≤ 0.1
        连续信号越长，说明信号更稳定
    '''

    def persistence1(self):
        signal_series = self._signal.fillna(0)
        max_consecutive = 0
        current_consecutive = 1
        for i in range(1, len(signal_series)):
            if signal_series.iloc[i] == signal_series.iloc[
                    i - 1] and signal_series.iloc[i] != 0:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1

        max_consecutive = max(max_consecutive, current_consecutive)
        # 归一化到[0,1]区间
        persistence = min(max_consecutive / len(signal_series), 1.0)
        return persistence

    '''
    风险型控制指标 -- 信号错误损失(信号方向错误时的平均损失)
    阈值（越低越好）：
        优秀：< 0.01
        良好：< 0.02
        及格：< 0.03
        不及格：≥ 0.03
        信号错误时的平均损失越小越安全。
    '''

    def error_loss(self):
        correct = (self._signal[self._mask] *
                   self._future_return[self._mask]) > 0
        if (~correct).sum() > 0:
            error_loss = abs(self._future_return[self._mask][~correct]).mean()
        else:
            error_loss = 0.0
        return error_loss

    '''
        风险型控制指标 -- 信号错误波动率（信号错误时损失的波动率）
        阈值（越低越好）：
            优秀：< 0.01
            良好：< 0.02
            及格：< 0.03
            不及格：≥ 0.03
            信号错误时损失的波动率越小越安全。
    '''

    def error_volatility(self):
        correct = (self._signal[self._mask] *
                   self._future_return[self._mask]) > 0
        if (~correct).sum() > 0:
            error_volatility = abs(
                self._future_return[self._mask][~correct]).std()
        else:
            error_volatility = 0.0
        return error_volatility

    '''
    最大连续错误次数
    阈值（越低越好，归一化到[0,1]）：
        优秀：< 0.1
        良好：< 0.2
        及格：< 0.3
        不及格：≥ 0.3
        连续错误次数越多，风险越大。
    '''

    def consecutive_errors(self):
        correct = (self._signal[self._mask] *
                   self._future_return[self._mask]) > 0
        max_consecutive_errors = 0
        current_consecutive_errors = 0
        for is_correct in correct:
            if not is_correct:
                current_consecutive_errors += 1
                max_consecutive_errors = max(max_consecutive_errors,
                                             current_consecutive_errors)
            else:
                current_consecutive_errors = 0
        max_consecutive_errors = min(max_consecutive_errors / len(correct),
                                     1.0)
        return max_consecutive_errors

    '''
    信号覆盖风险(信号缺失比例)/ 信号方向风险（信号方向变化频率）
    覆盖风险（risk()返回的coverage_risk, 越低越好）
        阈值：
            优秀：< 0.2
            良好：< 0.4
            及格：< 0.6
            不及格：≥ 0.6
            信号缺失比例越低越好。
    方向风险（risk()返回的direction_risk, 越低越好）
        阈值：
            优秀：< 0.1
            良好：< 0.2
            及格：< 0.3
            不及格：≥ 0.3
            信号方向变化频率越低越好。

    '''

    def risk(self):
        coverage_risk = 1 - self._mask.mean()
        signal_direction = np.sign(self._signal[self._mask])
        direction_changes = (signal_direction
                             != signal_direction.shift(1)).sum()
        direction_risk = direction_changes / len(
            self._signal[self._mask]) if len(
                self._signal[self._mask]) > 0 else 0
        return coverage_risk, direction_risk

    def discrete(self):
        metrics = {}
        ## . 预测能力指标
        # 命中率
        hit_rate = self.hit_rate()
        # 胜率盈亏比
        win_rate, profit_factor = self.win_loss()
        ### RankIC 及其统计显著性
        rank_ic, rank_ic_pvalue = self.rankic1()
        ### 分组收益
        group_return = self.group_return(n_groups=3)

        ## 交易效率指标
        ### 信号稳定性
        stability = self.stability()
        ### 信号换手率
        turnover = self.turnover()
        ### 信号一致性
        consistency = self.consistency()
        ### 信号连续性
        persistence = self.persistence1()

        ###
        pdb.set_trace()
        error_loss = self.error_loss()

        error_volatility = self.error_volatility()

        consecutive_errors = self.consecutive_errors()

        coverage_risk, direction_risk = self.risk()

        metrics['hit_rate'] = hit_rate
        metrics['win_rate'] = win_rate
        metrics['profit_factor'] = profit_factor
        metrics['rank_ic'] = rank_ic
        metrics['rank_ic_pvalue'] = rank_ic_pvalue
        metrics.update(group_return)

        metrics['stability'] = stability
        metrics['turnover'] = turnover
        metrics['consistency'] = consistency
        metrics['persistence'] = persistence

        metrics['error_loss'] = error_loss
        metrics['error_volatility'] = error_volatility
        metrics['consecutive_errors'] = consecutive_errors
        metrics['coverage_risk'] = coverage_risk
        metrics['direction_risk'] = direction_risk
        return metrics
