import pandas as pd
import numpy as np
import pdb
'''
scale_method: str = 'roll_min_max':
作用: 指定对因子进行放缩的方法。这是一个字符串，它有以下几种可选值，并且都严格遵守“无未来数据”原则：
'roll_min_max': 滚动窗口内的 Min-Max 放缩，将因子值放缩到 [-1, 1] 之间。
'roll_zscore': 滚动窗口内的 Z-score 放缩，将因子值转换为标准差单位，并截断（clip）到 [-3, 3] 后归一化到 [-1, 1]。
'roll_quantile': 滚动窗口内的分位数放缩，使用 25% 和 75% 分位数（四分位距）进行放缩，也映射到 [-1, 1]，对极端值更鲁棒。
'ew_zscore': 基于指数加权移动平均 (EWM) 的 Z-score 放缩，对近期数据赋予更高权重。
'train_const': 使用前 roll_win 个样本的均值和标准差作为固定参数来放缩整个时间序列的因子。这意味着在初始窗口之后，放缩参数是常量，不再滚动变化。
'''


class FactorEvaluate(object):

    def __init__(self,
                 factor_data: pd.DataFrame,
                 factor_name: str = 'factor',
                 ret_name: str = 'ret',
                 roll_win: int = 252,
                 fee: float = 0.0003,
                 scale_method: str = 'roll_min_max'):

        self.factor_data = factor_data.copy()
        self.factor_name = factor_name
        self.ret_name = ret_name
        self.roll_win = roll_win
        self.fee = fee
        self.scale_method = scale_method
        self.stats = None
        self._init_factor()

    def _init_factor(self):
        print('init factor')
        self.factor_data = self.factor_data.set_index('trade_time').sort_index()[[
            self.factor_name, self.ret_name]]

        '''
        self.factor_data['timestamp'] = pd.to_datetime(
            self.factor_data['date'].astype(str) + ' ' +
            self.factor_data['minTime'].astype(str))
        self.factor_data = self.factor_data.set_index('timestamp').sort_index()[[
            self.factor_name, self.ret_name
        ]]
        '''

    def _scale(self):
        x = self.factor_data[self.factor_name]
        win = self.roll_win
        #roll_min_max:
        #rmin = x.rolling(win).min(): 计算滚动窗口内的最小值。
        #rmax = x.rolling(win).max(): 计算滚动窗口内的最大值。
        #self.factor_df['f_scaled'] = 2 * (x - rmin) / (rmax - rmin).clip(lower=1e-8) - 1: 将因子值线性映射到 [-1, 1] 之间。.clip(lower=1e-8) 是为了防止分母为零而导致计算错误。
        if self.scale_method == 'roll_min_max':
            rmin = x.rolling(win).min()
            rmax = x.rolling(win).max()
            self.factor_data['f_scaled'] = 2 * \
                (x - rmin) / (rmax - rmin).clip(lower=1e-8) - 1

        # mu = x.rolling(win).mean(): 计算滚动窗口内的均值。
        # sg = x.rolling(win).std(): 计算滚动窗口内的标准差。
        # self.factor_df['f_scaled'] = ((x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3: 计算滚动 Z-score，并将其值截断到 [-3, 3] 范围内，然后除以3，使其结果也归一化到 [-1, 1]。
        elif self.scale_method == 'roll_zscore':
            mu = x.rolling(win).mean()
            sg = x.rolling(win).std()
            self.factor_data['f_scaled'] = (
                (x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3

        # roll_quantile:
        # q25 = x.rolling(win).quantile(0.25): 计算滚动窗口内的 25% 分位数。
        # q75 = x.rolling(win).quantile(0.75): 计算滚动窗口内的 75% 分位数。
        # self.factor_df['f_scaled'] = 2 * (x - q25) / (q75 - q25).clip(lower=1e-8) - 1: 基于四分位距进行放缩，映射到 [-1, 1]。
        elif self.scale_method == 'roll_quantile':
            q25 = x.rolling(win).quantile(0.25)
            q75 = x.rolling(win).quantile(0.75)
            self.factor_data['f_scaled'] = 2 * \
                (x - q25) / (q75 - q25).clip(lower=1e-8) - 1

        #ew_zscore:
        #ema = x.ewm(span=win, adjust=False).mean(): 计算指数加权移动平均。
        #evar = x.ewm(span=win, adjust=False).var(): 计算指数加权移动方差。
        #self.factor_df['f_scaled'] = ((x - ema) / np.sqrt(evar).clip(lower=1e-8)).clip(-3, 3) / 3: 基于指数加权统计量计算 Z-score，并进行截断和归一化。
        elif self.scale_method == 'ew_zscore':
            ema = x.ewm(span=win, adjust=False).mean()
            evar = x.ewm(span=win, adjust=False).var()
            self.factor_data['f_scaled'] = (
                (x - ema) / np.sqrt(evar).clip(lower=1e-8)).clip(-3, 3) / 3

        #train_const:
        #mu = x.iloc[:win].mean(): 使用前 win 个（即训练集）样本的均值。
        #sg = x.iloc[:win].std(): 使用前 win 个样本的标准差。
        #self.factor_df['f_scaled'] = ((x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3: 使用固定的均值和标准差对所有因子值进行 Z-score 放缩，然后截断和归一化。
        elif self.scale_method == 'train_const':
            # 用前 roll 个样本做训练集
            mu = x.iloc[:win].mean()
            sg = x.iloc[:win].std()
            self.factor_data['f_scaled'] = (
                (x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3
        else:
            raise ValueError('Unknown scale_method')

    def cal_ic(self):
        """
        计算因子与预期收益的滚动相关性
        """
        self.factor_data['ic'] = self.factor_data[self.ret_name].rolling(
            window=self.roll_win,
            min_periods=5).corr(self.factor_data[self.factor_name])

        return {
            'ic_mean': self.factor_data['ic'].mean(),
            'ic_std': self.factor_data['ic'].std(),
            'ic_ir': self.factor_data['ic'].mean() /
            self.factor_data['ic'].std()  # 衡量因子预测能力的稳定性和质量。
        }

    def cal_pnl(self):
        self.factor_data['pos'] = self.factor_data['f_scaled'] # 将放缩后的因子值 f_scaled 直接作为每期的交易头寸。例如，f_scaled 为 0.5 可能代表 50% 的多头头寸，-0.8 代表 80% 的空头头寸。
        self.factor_data['gross_ret'] = self.factor_data['pos'] * \
            self.factor_data[self.ret_name] # 计算每期的总收益（未扣除费用），即头寸乘以对应期的远期收益。
        self.factor_data['turnover'] = np.abs(
            np.diff(self.factor_data['pos'], prepend=0)) # 计算每期的换手率。它衡量的是头寸的绝对变化。np.diff() 计算连续差值，prepend=0 用于处理第一个头寸的换手率（假设初始头寸为0）。
        self.factor_data['net_ret'] = (self.factor_data['gross_ret'] -
                                     self.fee * self.factor_data['turnover']) # 计算每期的净收益，即从总收益中减去交易费用。费用是换手率乘以设定的 fee。
        self.factor_data['nav'] = (1 + self.factor_data['net_ret']).cumprod() #  计算净值曲线（Net Asset Value）。这是 (1 + 净收益) 的累积乘积，代表了投资组合的模拟价值随时间的变化。

        # -------- 基础统计 --------
        total_ret = self.factor_data['nav'].iloc[-1] - 1  # 累计收益 整个回测期间的累计收益。
        avg_ret = self.factor_data['net_ret'].mean()  # 平均每次交易收益 
        max_dd = (self.factor_data['nav'] / self.factor_data['nav'].cummax() -
                  1).min() # 找到历史最高净值，然后计算当前净值相对历史最高点的最大下跌百分比。
        calmar = total_ret / abs(max_dd) if max_dd != 0 else np.nan # 卡玛比率

        turnover = self.factor_data['turnover'].mean() # 平均每期换手率。
        win_rate = (self.factor_data['net_ret'] > 0).mean() # 胜率，即净收益为正的周期所占的比例。
        profit_ratio = (
            self.factor_data.loc[self.factor_data['net_ret'] > 0, 'net_ret'].sum()
            / np.abs(self.factor_data.loc[self.factor_data['net_ret'] < 0,
                                        'net_ret']).sum()) #  盈亏比。正收益的绝对值之和除以负收益的绝对值之和。衡量盈利时的平均盈利幅度与亏损时的平均亏损幅度之比。

        return {
            'total_ret': total_ret,
            'avg_ret': avg_ret,
            'max_dd': max_dd,
            'calmar': calmar,
            'turnover': turnover,
            'win_rate': win_rate,
            'profit_ratio': profit_ratio,
        }

    def run(self):
        self._scale()
        ic_stats = self.cal_ic()  # 计算ic指标
        pnl_stats = self.cal_pnl() # 计算交易绩效
        pnl_stats.update(ic_stats)
        return pnl_stats
