import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  #
import seaborn as sns
'''
scale_method: str = 'roll_min_max':
作用: 指定对因子进行放缩的方法。这是一个字符串，它有以下几种可选值，并且都严格遵守“无未来数据”原则：
'roll_min_max': 滚动窗口内的 Min-Max 放缩，将因子值放缩到 [-1, 1] 之间。
'roll_zscore': 滚动窗口内的 Z-score 放缩，将因子值转换为标准差单位，并截断（clip）到 [-3, 3] 后归一化到 [-1, 1]。
'roll_quantile': 滚动窗口内的分位数放缩，使用 25% 和 75% 分位数（四分位距）进行放缩，也映射到 [-1, 1]，对极端值更鲁棒。
'ew_zscore': 基于指数加权移动平均 (EWM) 的 Z-score 放缩，对近期数据赋予更高权重。
'train_const': 使用前 roll_win 个样本的均值和标准差作为固定参数来放缩整个时间序列的因子。这意味着在初始窗口之后，放缩参数是常量，不再滚动变化。
'''

# 建议设置一个美观的绘图风格
sns.set_style('whitegrid')


class FactorEvaluate1(object):

    def __init__(self,
                 factor_data: pd.DataFrame,
                 factor_name: str = 'factor',
                 ret_name: str = 'ret',
                 roll_win: int = 252,
                 fee: float = 0.0003,
                 scale_method: str = 'roll_min_max',
                 annualization_factor: int = 252,
                 expression=None,
                 name=None):

        self.factor_data = factor_data.copy()
        self.factor_name = factor_name
        self.ret_name = ret_name
        self.roll_win = roll_win
        self.fee = fee
        self.scale_method = scale_method
        self.annualization_factor = annualization_factor
        self.name = name
        self.expression = expression
        self.stats = None
        self._init_factor()

    def _init_factor(self):
        self.factor_data['trade_time'] = pd.to_datetime(
            self.factor_data['trade_time'])
        self.factor_data.set_index('trade_time', inplace=True)
        self.factor_data = self.factor_data.sort_index()[[
            self.factor_name, self.ret_name
        ]]

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
            
        elif self.scale_method == 'raw':
            # 直接使用原始值，不进行任何缩放，假设为已经处理好的因子值，离散值为[-1,0,1], 连续值为[-1，1]
            self.factor_data['f_scaled'] = x
        else:
            raise ValueError('Unknown scale_method')

    def cal_ic(self):
        """
        计算因子与预期收益的滚动相关性
        """
        self.factor_data['ic'] = self.factor_data[self.ret_name].rolling(
            window=self.roll_win,
            min_periods=5).corr(self.factor_data[self.factor_name])

        self.factor_data['cumsum_ic'] = self.factor_data['ic'].cumsum()
        ic_mean = self.factor_data['ic'].mean()
        ic_std = self.factor_data['ic'].std()
        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_mean / ic_std if ic_std != 0 else 0  # 衡量因子预测能力的稳定性和质量。
        }

    def cal_pnl(self):
        self.factor_data['pos'] = self.factor_data[
            'f_scaled']  # 将放缩后的因子值 f_scaled 直接作为每期的交易头寸。例如，f_scaled 为 0.5 可能代表 50% 的多头头寸，-0.8 代表 80% 的空头头寸。

        self.factor_data[
            'gross_ret'] = self.factor_data['pos'] * self.factor_data[
                self.ret_name]  # 计算每期的总收益（未扣除费用），即头寸乘以对应期的远期收益。

        self.factor_data['turnover'] = np.abs(
            np.diff(self.factor_data['pos'], prepend=0)
        )  # 计算每期的换手率。它衡量的是头寸的绝对变化。np.diff() 计算连续差值，prepend=0 用于处理第一个头寸的换手率（假设初始头寸为0）。

        self.factor_data['net_ret'] = (
            self.factor_data['gross_ret'] -
            self.fee * self.factor_data['turnover']
        )  # 计算每期的净收益，即从总收益中减去交易费用。费用是换手率乘以设定的 fee。

        self.factor_data['nav'] = (1 + self.factor_data['net_ret']).cumprod(
        )  #  计算净值曲线（Net Asset Value）。这是 (1 + 净收益) 的累积乘积，代表了投资组合的模拟价值随时间的变化。

        # -------- 基础统计 --------
        total_ret = self.factor_data['nav'].iloc[-1] - 1  # 累计收益 整个回测期间的累计收益。

        avg_ret = self.factor_data['net_ret'].mean()  # 平均每次交易收益

        max_dd = (self.factor_data['nav'] / self.factor_data['nav'].cummax() -
                  1).min()  # 找到历史最高净值，然后计算当前净值相对历史最高点的最大下跌百分比。

        calmar = total_ret / abs(max_dd) if max_dd != 0 else np.nan  # 卡玛比率

        ## 换算日收益率 算夏普
        daily_net_ret = self.factor_data['net_ret'].resample('1D').agg(
            {'net_ret': 'sum'})

        rets_mean = daily_net_ret['net_ret'].mean() * 250
        rets_std = daily_net_ret['net_ret'].std() * np.sqrt(250)
        sharpe2 = rets_mean / rets_std if rets_std != 0 else 0
        sharpe1 = self.factor_data['net_ret'].mean() / self.factor_data[
            'net_ret'].std() if self.factor_data['net_ret'].std() != 0 else 0

        turnover = self.factor_data['turnover'].mean()  # 平均每期换手率。

        win_rate = (self.factor_data['net_ret']
                    > 0).mean()  # 胜率，即净收益为正的周期所占的比例。

        profit_sum = self.factor_data.loc[self.factor_data['net_ret'] > 0,
                                          'net_ret'].sum()
        loss_sum = np.abs(self.factor_data.loc[self.factor_data['net_ret'] < 0,
                                               'net_ret']).sum()
        profit_ratio = profit_sum / loss_sum if loss_sum != 0 else np.inf  #  盈亏比。正收益的绝对值之和除以负收益的绝对值之和。衡量盈利时的平均盈利幅度与亏损时的平均亏损幅度之比。

        return {
            'total_ret': total_ret,
            'avg_ret': avg_ret,
            'max_dd': max_dd,
            'calmar': calmar,
            'sharpe1': sharpe1,
            'sharpe2': sharpe2,
            'turnover': turnover,
            'win_rate': win_rate,
            'profit_ratio': profit_ratio
        }

    def _cal_autocorr(self):
        """计算因子和收益率的滞后1期自相关性。"""
        factor_ac = self.factor_data[self.factor_name].autocorr(lag=1)
        ret_ac = self.factor_data[self.ret_name].autocorr(lag=1)
        return {'factor_autocorr': factor_ac, 'ret_autocorr': ret_ac}

    def _check_warnings(self):
        """检查关键指标并打印警告。"""
        print("\n--- Sanity Checks & Warnings ---")
        # 1. 检查收益率自相关性
        ret_ac = self.stats.get('ret_autocorr', 0)
        if not (-0.1 < ret_ac < 0.1):
            print(
                f"⚠️  WARNING: Return autocorrelation is {ret_ac:.3f}. Normal range is [-0.1, 0.1]. High value might indicate data issues (e.g., stale prices)."
            )
        else:
            print(f"✅ Return autocorrelation ({ret_ac:.3f}) is normal.")

        # 2. 检查因子自相关性 (极端值)
        factor_ac = self.stats.get('factor_autocorr', 0)
        if abs(factor_ac) > 0.99:
            print(
                f"⚠️  WARNING: Factor autocorrelation is {factor_ac:.3f}, which is extremely high. The factor is nearly non-stationary and may have high turnover."
            )
        else:
            print(
                f"✅ Factor autocorrelation ({factor_ac:.3f}) is within a reasonable range."
            )

        # 3. 检查ICIR
        ic_ir = self.stats.get('ic_ir', 0)
        if ic_ir < 0.3:
            print(
                f"⚠️  WARNING: ICIR is {ic_ir:.3f}, which is low. Factor's predictive power is unstable."
            )
        else:
            print(f"✅ ICIR ({ic_ir:.3f}) indicates stable performance.")

    def run(self, is_check=False):
        self._scale()
        ic_stats = self.cal_ic()
        if ic_stats['ic_mean'] < 0:
            self.factor_data['f_scaled'] *= -1
            #ic_stats = self.cal_ic()
            if is_check:
                print("INFO: IC Mean is negative. Factor has been inverted.")

        pnl_stats = self.cal_pnl()
        autocorr_stats = self._cal_autocorr()  # 计算自相关性

        # 合并所有统计数据
        pnl_stats.update(ic_stats)
        pnl_stats.update(autocorr_stats)

        self.stats = pnl_stats
        if is_check:
            self._check_warnings()  # 运行警告检查
        return self.stats

    def plot_results(self):
        if self.stats is None:
            raise RuntimeError(
                "Please run the 'run()' method before plotting.")

        def set_sequential_xticks(ax, series, num_ticks=7):
            """
            为一个使用整数索引绘图的坐标轴设置日期标签。
            ax: a matplotlib axis object.
            series: The original pandas Series with a DatetimeIndex.
            num_ticks: The desired number of date labels on the x-axis.
            """
            # 计算刻度的整数位置
            tick_positions = np.linspace(0, len(series) - 1, num_ticks, dtype=int)
            # 获取这些位置对应的日期标签
            tick_labels = [series.index[i].strftime('%Y-%m-%d') for i in tick_positions]
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=30, ha='right')


        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle(
            f"Factor Evaluation: {self.factor_name} vs {self.ret_name}",
            fontsize=18)

        # 1. 净值曲线 (NAV)
        ax1 = axes[0, 0]
        nav_data = self.factor_data['nav'].dropna()
        gross_ret_data = (1 + self.factor_data['gross_ret']).cumprod().dropna()

        # 使用 use_index=False 来忽略时间轴，绘制连续序列
        nav_data.plot(ax=ax1, label='Net Asset Value (NAV)', color='blue', use_index=False)
        gross_ret_data.plot(ax=ax1, label='Cumulative Gross Return', color='orange', linestyle='--', use_index=False)
        
        # 使用辅助函数设置X轴标签
        set_sequential_xticks(ax1, nav_data)
        
        ax1.set_title("Performance")
        ax1.set_ylabel("NAV")
        ax1.set_xlabel("trade_time (sequential)") # 标签提示X轴是序列
        ax1.legend()
        ax1.grid(True)




        # 2. 绩效指标表格
        ax_table = axes[0, 1]
        ax_table.axis('off')

        report_parts = []
        if self.expression is not None:
            report_parts.append(f"Expression: {self.expression}")
        if self.name is not None:
            report_parts.append(f"Name: {self.name}")

        if len(report_parts) > 0:
            report_parts.append("\n")

        performance_metrics = (
            f"--- Performance Metrics ---\n"
            f"{'Avg Return':<20}: {(self.stats.get('avg_ret', float('nan')) * 10000):.2f} bps\n"
            f"{'Total Return':<20}: {self.stats['total_ret']:.2%}\n"
            f"{'Sharpe Ratio':<20}: {self.stats['sharpe1']:.2f}\n"
            f"{'Ann Sharpe Ratio':<20}: {self.stats['sharpe2']:.2f}\n"
            f"{'Max Drawdown':<20}: {self.stats['max_dd']:.2%}\n"
            f"{'Calmar Ratio':<20}: {self.stats.get('calmar', float('nan')):.2f}\n"
            f"{'Win Rate':<20}: {self.stats['win_rate']:.2%}\n"
            f"{'Profit/Loss Ratio':<20}: {self.stats['profit_ratio']:.2f}\n"
            f"\n--- Factor Characteristics ---\n"
            f"{'IC Mean':<20}: {self.stats['ic_mean']:.4f}\n"
            f"{'ICIR':<20}: {self.stats['ic_ir']:.4f}\n"
            f"{'Mean Turnover':<20}: {self.stats['turnover']:.4f}\n"
            f"{'Factor Autocorr':<20}: {self.stats['factor_autocorr']:.4f}\n"  # 新增
            f"{'Return Autocorr':<20}: {self.stats['ret_autocorr']:.4f}\n"  # 新增
        )
        report_parts.append(performance_metrics)
        stats_text = "\n".join(report_parts)

        ax_table.text(0.05,
                      0.95,
                      stats_text,
                      transform=ax_table.transAxes,
                      fontsize=12,
                      verticalalignment='top',
                      fontfamily='monospace')
        ax_table.set_title("Key Performance Indicators", fontsize=14)

        # 3. IC 和 累计IC
        ax3 = axes[1, 0]
        ic_data = self.factor_data['ic'].dropna()
        cumsum_ic_data = self.factor_data['cumsum_ic'].dropna()
        
        ic_data.plot(ax=ax3, label='Rolling IC', color='steelblue', alpha=0.8, use_index=False)
        set_sequential_xticks(ax3, ic_data)
        
        ax3.set_ylabel("Rolling IC", color='steelblue')
        ax3_twin = ax3.twinx()
        
        cumsum_ic_data.plot(ax=ax3_twin, label='Cumulative IC', color='black', linestyle='--', use_index=False)
        
        ax3_twin.set_ylabel("Cumulative IC", color='black')
        ax3.set_title("IC Analysis")
        ax3.set_xlabel("trade_time (sequential)")
        ax3.axhline(0, color='red', linestyle='--', linewidth=1)
        ax3.grid(True)



        # 4. 因子 vs. 收益率散点图
        ax4 = axes[1, 1]
        sns.scatterplot(x=self.factor_name,
                        y=self.ret_name,
                        data=self.factor_data,
                        ax=ax4,
                        s=10,
                        alpha=0.3,
                        color='purple')
        ax4.set_title("Factor vs. Return Scatter Plot")
        ax4.set_xlabel("Original Factor Value")
        ax4.set_ylabel("Forward Return")
        ax4.grid(True)

        # 5. 每日收益率与回撤
        ax5 = axes[2, 0]
        drawdown_data = ((self.factor_data['nav'] / self.factor_data['nav'].cummax() - 1) * 100).dropna()
        
        drawdown_data.plot(ax=ax5, color='red', alpha=0.8, use_index=False)
        # fill_between 需要 numpy 数组
        ax5.fill_between(np.arange(len(drawdown_data)), drawdown_data.values, 0, color='red', alpha=0.2)
        set_sequential_xticks(ax5, drawdown_data)
        
        ax5.set_title(f"Drawdown Over Time (Max = {self.stats['max_dd']:.2%})")
        ax5.set_ylabel("Drawdown (%)")
        ax5.set_xlabel("trade_time (sequential)")
        ax5.set_ylim(bottom=None, top=0.5)
        ax5.grid(True)



        # 6. 换手率时序图
        ax6 = axes[2, 1]
        turnover_data = self.factor_data['turnover'].dropna()
        
        turnover_data.plot(ax=ax6, color='teal', use_index=False)
        set_sequential_xticks(ax6, turnover_data)
        
        ax6.set_title(
            f"Turnover Over Time (Mean = {self.stats['turnover']:.3f})")
        ax6.set_ylabel("Turnover")
        ax6.set_xlabel("trade_time (sequential)")
        ax6.grid(True)

        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter('%Y-%m')


        # 2. 将格式应用到所有需要时间轴的子图
        #for ax in [ax1, ax3, ax5, ax6]:
        #    ax.xaxis.set_major_locator(locator)
        #    ax.xaxis.set_major_formatter(formatter)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()
