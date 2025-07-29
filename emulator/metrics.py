import pdb
import pandas as pd
import numpy as np

class Metrics(object):
    """
    根据每日净值序列计算并展示策略的各项绩效指标。
    """
    def __init__(self, daily_results, initial_capital):
        if not daily_results:
            self.daily_df = pd.DataFrame()
        else:
            self.daily_df = pd.DataFrame(daily_results).set_index('date')
        
        self.initial_capital = initial_capital
        self.final_net_value = self.daily_df['net_value'].iloc[-1] if not self.daily_df.empty else initial_capital
        self.total_commission = self.daily_df['commission'].iloc[-1] if not self.daily_df.empty else 0
        
    def calculate_metrics(self):
        """计算所有核心绩效指标。"""
        if self.daily_df.empty:
            print("No trades were made. No performance to analyze.")
            return {}

        # 使用实例属性 self.daily_df
        df = self.daily_df
        
        # --- 核心修正：正确计算每日收益率 ---
        initial_row = pd.DataFrame([{'net_value': self.initial_capital}], index=[df.index[0] - pd.Timedelta(days=1)])
        net_value_with_initial = pd.concat([initial_row['net_value'], df['net_value']])
        
        daily_returns = net_value_with_initial.pct_change().iloc[1:]
        df['daily_return'] = daily_returns

        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

        # --- 核心修正：正确计算最大回撤 ---
        drawdown_series = pd.concat([pd.Series([self.initial_capital]), df['net_value']])
        cumulative_max = drawdown_series.cummax()
        drawdown = (drawdown_series / cumulative_max - 1).iloc[1:]
        max_drawdown = drawdown.min()
        
        # 计算年化收益和夏普比率
        total_return = (self.final_net_value / self.initial_capital) - 1
        annualized_return = total_return * (252 / len(df)) if len(df) > 0 else 0
        sharpe_ratio = (df['daily_return'].mean() / df['daily_return'].std()) * np.sqrt(252) if df['daily_return'].std() != 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def print_summary(self):
        """打印绩效总结报告。"""
        metrics = self.calculate_metrics()
        
        print("\n--- Strategy Performance Analysis (Corrected) ---")
        print(f"Initial Capital: {self.initial_capital:,.2f}")
        print(f"Final Net Value: {self.final_net_value:,.2f}")
        
        if not metrics:
            return

        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Total Commission Paid: {self.total_commission:,.2f}")

        try:
            import matplotlib.pyplot as plt
            # --- 核心修正：使用 self.daily_df 替代 df ---
            if not self.daily_df.empty: # 增加一个检查，确保 DataFrame 不为空
                plot_series = pd.concat([
                    pd.Series({self.daily_df.index[0] - pd.Timedelta(days=1): self.initial_capital}), 
                    self.daily_df['net_value']
                ])
                plot_series.plot(title='Portfolio Net Value Over Time', grid=True, marker='o', linestyle='-')
                plt.ylabel('Net Value')
                plt.xlabel('Date')
                plt.show()
        except ImportError:
            print("\nMatplotlib not installed. Skipping plot generation.")