import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  #
import seaborn as sns
import ultron.factor.empyrical as empyrical
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret


class StrategyEvaluate1(object):

    def __init__(self,
                 pos_data: pd.DataFrame,
                 total_data: pd.DataFrame,
                 strategy_settings: dict,
                 strategy_name: str,
                 ret_name: str,
                 annualization_factor: int = 252):
        """
        初始化评估器。

        Args:
            daily_returns_df (pd.DataFrame): 包含每日净收益率 ('daily_net_ret') 的 DataFrame。
            name (str, optional): 策略名称.
            expression (str, optional): 策略表达式.
        """
        self.pos_data = pos_data.copy()
        self.total_data = total_data.copy()
        self.strategy_settings = strategy_settings
        self.strategy_name = strategy_name
        self.ret_name = ret_name
        self.annualization_factor = annualization_factor
        self.stats = None

    def timesequence_returns(self):
        min_returns = calculate_ful_ts_ret(
            pos_data=self.pos_data,
            total_data=self.total_data,
            strategy_settings=self.strategy_settings,
            ret_name=self.ret_name,
            agg=False)
        daily_returns = min_returns.resample('1D').agg({
            'a_ret': 'sum',
            'n_ret': 'sum',
        }).dropna().fillna({
            'a_ret': 0,
            'n_ret': 0,
        }).fillna(method='ffill')
        return daily_returns, min_returns

    def run(self, agg=False):
        ## 时序收益
        daily_returns, min_returns = self.timesequence_returns()
        #self.daily_returns = daily_returns  #['a_ret']
        self.min_returns = min_returns  #['a_ret']
      
        turnover = np.abs(
            np.diff(self.pos_data.values.reshape(-1, ), prepend=0)).mean()

        max_drawdown = empyrical.max_drawdown(returns=min_returns['a_ret'])
        final_return = empyrical.cum_returns_final(
            returns=min_returns['a_ret'])

        annual_return = empyrical.annual_return(returns=daily_returns['a_ret'],
                                                period=empyrical.DAILY)
        annual_volatility = empyrical.annual_volatility(
            returns=daily_returns['a_ret'], period=empyrical.DAILY)
        downside_risk = empyrical.downside_risk(returns=daily_returns['a_ret'],
                                                period=empyrical.DAILY)

        sharpe_ratio = empyrical.sharpe_ratio(returns=daily_returns['a_ret'],
                                              period=empyrical.DAILY)
        '''
        sharpe_ratio = empyrical.sharpe_ratio(returns=daily_returns['a_ret'],
                                              period=empyrical.DAILY)
        calmar_ratio = empyrical.calmar_ratio(returns=daily_returns['a_ret'],
                                              period=empyrical.DAILY)

        annual_return = empyrical.annual_return(returns=daily_returns['a_ret'],
                                                period=empyrical.DAILY)
        annual_volatility = empyrical.annual_volatility(
            returns=daily_returns['a_ret'], period=empyrical.DAILY)
        downside_risk = empyrical.downside_risk(returns=daily_returns['a_ret'],
                                                period=empyrical.DAILY)
        '''
        final_return = empyrical.cum_returns_final(
            returns=min_returns['a_ret'])
        calmar_ratio = final_return / abs(
            max_drawdown) if max_drawdown != 0 else np.nan

        win_ratio = empyrical.win_ratio(min_returns['a_ret'])

        profit_sum = min_returns['a_ret'][min_returns['a_ret'] > 0].sum()
        loss_sum = np.abs(min_returns['a_ret'][min_returns['a_ret'] < 0].sum())
        profit_loss_ratio = profit_sum / loss_sum if loss_sum != 0 else np.inf

        self.stats = {
            'final_return': final_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'p/l_ratio': profit_loss_ratio,
            'win_ratio': win_ratio,
            'turnover': turnover,
            'max_drawdown': max_drawdown,
            'downside_risk': downside_risk,
            'annual_volatility': annual_volatility
        }
        return self.stats

    def plot_results(self):
        if self.stats is None:
            raise RuntimeError(
                "Please run the 'run()' method before plotting.")

        def set_sequential_xticks(ax, series, num_ticks=7):
            tick_positions = np.linspace(0,
                                         len(series) - 1,
                                         num_ticks,
                                         dtype=int)
            tick_labels = [
                series.index[i].strftime('%Y-%m-%d') for i in tick_positions
            ]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=30, ha='right')

        fig, axes = plt.subplots(2,
                                 2,
                                 figsize=(18, 12),
                                 constrained_layout=True)
        fig.suptitle(
            f"Strategy Evaluation  {self.strategy_name} vs {self.ret_name}",
            fontsize=18)

        # 1. 收益率曲线 (PnL Curve)
        ax1 = axes[0, 0]
        nav_actual = (1 + self.min_returns['a_ret']).cumprod()
        nav_gross = (1 + self.min_returns['n_ret']).cumprod()

        nav_actual.plot(ax=ax1,
                        label='Net Return (w/ fee)',
                        color='blue',
                        use_index=False)
        nav_gross.plot(ax=ax1,
                       label='Gross Return (w/o fee)',
                       color='orange',
                       linestyle='--',
                       use_index=False)
        set_sequential_xticks(ax1, nav_actual)
        ax1.set_title("1. PnL Curve (Based on Minute Returns)")
        ax1.set_ylabel("Net Asset Value (NAV)")
        ax1.set_xlabel("Trade Time (Sequential)")
        ax1.legend()
        ax1.grid(True)

        # 2. 绩效指标表格
        ax_table = axes[0, 1]
        ax_table.axis('off')

        report_parts = []
        #if self.expression is not None:
        #    report_parts.append(f"Expression: {self.expression}")

        if self.strategy_name is not None:
            report_parts.append(f"Strategy Name: {self.strategy_name}")

        if len(report_parts) > 0:
            report_parts.append("\n")

        performance_metrics = (
            f"--- Performance Metrics ---\n"
            f"{'Total Return':<20}: {self.stats['final_return']:.2%}\n"
            f"{'Annual Return':<20}: {self.stats['annual_return']:.2%}\n"
            f"{'Sharpe Ratio':<20}: {self.stats['sharpe_ratio']:.2f}\n"
            f"{'Calmar Ratio':<20}: {self.stats.get('calmar_ratio', float('nan')):.2f}\n"
            f"{'Profit/Loss Ratio':<20}: {self.stats['p/l_ratio']:.2f}\n"
            f"{'Win Ratio':<20}: {self.stats['win_ratio']:.2%}\n"
            f"{'Mean Turnover':<20}: {self.stats['turnover']:.4f}\n"
            f"{'Max Drawdown':<20}: {self.stats['max_drawdown']:.2%}\n"
            #f"{'Downside Risk':<20}: {self.stats['downside_risk']:.2%}\n"
            f"{'Annual Volatility':<20}: {self.stats['annual_volatility']:.2%}\n"
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
        ax_table.set_title("Key Metrics Indicators", fontsize=14)

        # 3. 回撤时序图 (Drawdown)
        ax3 = axes[1, 0]
        drawdown_data = (nav_actual / nav_actual.cummax() - 1) * 100
        drawdown_data.plot(ax=ax3,
                           color='red',
                           alpha=0.8,
                           use_index=False,
                           label='Drawdown')
        ax3.fill_between(np.arange(len(drawdown_data)),
                         drawdown_data.values,
                         0,
                         color='red',
                         alpha=0.2)
        set_sequential_xticks(ax3, drawdown_data)
        ax3.set_title(
            f"3. Drawdown Curve (Max = {self.stats['max_drawdown']:.2%})")
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_xlabel("Trade Time (Sequential)")
        ax3.grid(True)

        # 7. 信号与收益关系图
        ax4 = axes[1, 1]
        merged_df = pd.concat([
            self.pos_data.stack().droplevel(level=1), self.min_returns['a_ret']
        ],
                              axis=1).dropna()
        merged_df.columns = ['pos', 'ret']
        try:
            merged_df['pos_bin'] = pd.qcut(merged_df['pos'],
                                           q=10,
                                           duplicates='drop',
                                           labels=False)
            bin_labels_series = pd.qcut(merged_df['pos'],
                                        q=10,
                                        duplicates='drop')
            bin_labels = [
                f'{b.left:.2f} to {b.right:.2f}'
                for b in bin_labels_series.categories
            ]
            sns.boxplot(x='pos_bin',
                        y='ret',
                        data=merged_df,
                        ax=ax4,
                        palette='vlag')
            ax4.set_xticklabels(bin_labels, rotation=45, ha='right')
            ax4.set_xlabel("Position Bin (Quantiles)")
        except Exception:
            sns.scatterplot(x='pos',
                            y='ret',
                            data=merged_df,
                            ax=ax4,
                            alpha=0.2)
            ax4.set_xlabel("Position")
        ax4.axhline(0, color='black', linestyle='--', linewidth=1)
        ax4.set_title("7. Return vs. Previous Position")
        ax4.set_ylabel("Next Minute's Net Return")
        ax4.grid(True)

        # 4. 换手率
        '''
        turnover_series_min = self.pos_data.stack().diff().abs().fillna(0)
        print(turnover_series_min)
        turnover_series = turnover_series_min.droplevel(level=1).resample(
            'D').mean().fillna(0)

        ax4 = axes[1, 1]
        turnover_series.plot(ax=ax4, color='teal', use_index=False)
        set_sequential_xticks(ax4, turnover_series)
        ax4.set_title(
            f"4. Daily Turnover (Mean = {self.stats['Mean Daily Turnover']:.3f})"
        )
        ax4.set_ylabel("Daily Turnover")
        ax4.grid(True)
        '''
