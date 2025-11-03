import os, hashlib
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.dates as mdates  #
import matplotlib.pyplot as plt
import seaborn as sns

from kdutils.macro2 import *

from lumina.genetic.util import create_id
from lib.iux001 import aggregation_data
from lib.cux001 import FactorEvaluate1
from lib.aux001 import calc_expression


def generate_simple_id(formula: str) -> str:
    # 1. 移除空格并转为小写
    normalized_formula = formula.replace(" ", "").lower()

    # 2. 使用 MD5 哈希
    # .encode('utf-8') 是必须的，因为哈希函数处理的是字节
    hasher = hashlib.md5(normalized_formula.encode('utf-8'))

    # .hexdigest() 返回16进制的哈希字符串
    return hasher.hexdigest()


def load_programs(method,
                  instruments,
                  period,
                  task_id,
                  session,
                  category='gentic'):
    dirs = os.path.join(base_path, method, instruments, category, 'ic',
                        str(task_id), "nxt1_ret_{}h".format(str(period)),
                        str(session))
    filename = os.path.join(
        dirs, "programs_{0}_{1}.feather".format(str(task_id), str(session)))

    programs = pd.read_feather(filename)

    programs = programs[programs['final_fitness'] > 0.02][[
        'name', 'formual', 'final_fitness'
    ]]
    return programs


class FactorComparator:

    def __init__(self,
                 eval_left: FactorEvaluate1,
                 eval_right: FactorEvaluate1,
                 left_name: str,
                 right_name: str,
                 expression: str,
                 name: str = None):
        if eval_left.stats is None or eval_right.stats is None:
            raise ValueError(
                "Evaluation objects must be run before creating a comparator.")
        self.eval_left, self.eval_right = eval_left, eval_right
        self.left_name, self.right_name = left_name, right_name
        self.expression = expression
        self.name = name if isinstance(name, str) else create_id(
            generate_simple_id(expression))

    def _generate_stats_text(self) -> str:
        metrics = [('Avg Return (bps)', 'avg_ret', '.2f', 10000),
                   ('Total Return', 'total_ret', '.2%'),
                   ('Sharpe Ratio', 'sharpe1', '.2f'),
                   ('Ann Sharpe Ratio', 'sharpe2', '.2f'),
                   ('Max Drawdown', 'max_dd', '.2%'),
                   ('Calmar Ratio', 'calmar', '.2f'),
                   ('IC Mean', 'ic_mean', '.4f'), ('ICIR', 'ic_ir', '.4f'),
                   ('Mean Turnover', 'turnover', '.4f'),
                   ('Factor Autocorr', 'factor_autocorr', '.4f'),
                   ('Return Autocorr', 'ret_autocorr', '.4f')]
        text = f"Factor Comparison: {self.left_name} vs. {self.right_name}\nExpression: {self.expression}\n\n"
        text += f"{'Metric':<20} | {self.left_name:<15} | {self.right_name:<15}\n" + "-" * 55 + "\n"
        for name, key, fmt, *mult in metrics:
            val_l, val_r = self.eval_left.stats.get(
                key, np.nan), self.eval_right.stats.get(key, np.nan)
            m = mult[0] if mult else 1
            l_str = f"{val_l * m:{fmt}}" if pd.notna(val_l) else "N/A"
            r_str = f"{val_r * m:{fmt}}" if pd.notna(val_r) else "N/A"
            text += f"{name:<20} | {l_str:<15} | {r_str:<15}\n"
        return text

    def plot_comparison(self):
        sns.set_style('whitegrid')
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle(
            f"Factor Comparison: {self.left_name} vs. {self.right_name}\nExpression: {self.expression}\
                Name:{self.name}",
            fontsize=18)

        ax1 = axes[0, 0]
        self.eval_left.factor_data['nav'].dropna().plot(ax=ax1,
                                                        label=self.left_name,
                                                        color='blue')
        self.eval_right.factor_data['nav'].dropna().plot(ax=ax1,
                                                         label=self.right_name,
                                                         color='orange')
        ax1.set_title("Net Asset Value (NAV) Comparison")
        ax1.legend()
        ax1.grid(True)

        ax_table = axes[0, 1]
        ax_table.axis('off')
        ax_table.text(0.01,
                      0.95,
                      "\n".join(self._generate_stats_text().split('\n')[3:]),
                      transform=ax_table.transAxes,
                      fontsize=12,
                      va='top',
                      fontfamily='monospace')
        ax_table.set_title("Key Performance Indicators", fontsize=14)

        ax3 = axes[1, 0]
        self.eval_left.factor_data['cumsum_ic'].plot(ax=ax3,
                                                     label=self.left_name,
                                                     color='blue')
        self.eval_right.factor_data['cumsum_ic'].plot(ax=ax3,
                                                      label=self.right_name,
                                                      color='orange')
        ax3.set_title("Cumulative IC Comparison")
        ax3.legend()
        ax3.grid(True)

        ax4 = axes[1, 1]
        sns.kdeplot(data=self.eval_left.factor_data['ic'].dropna(),
                    ax=ax4,
                    label=self.left_name,
                    color='blue',
                    fill=True)
        sns.kdeplot(data=self.eval_right.factor_data['ic'].dropna(),
                    ax=ax4,
                    label=self.right_name,
                    color='orange',
                    fill=True)
        ax4.set_title("Rolling IC Distribution")
        ax4.legend()
        ax4.grid(True)

        ax5 = axes[2, 0]
        dd_left = (self.eval_left.factor_data['nav'] /
                   self.eval_left.factor_data['nav'].cummax() - 1)
        dd_right = (self.eval_right.factor_data['nav'] /
                    self.eval_right.factor_data['nav'].cummax() - 1)
        dd_left.plot(ax=ax5, label=self.left_name, color='blue', alpha=0.8)
        dd_right.plot(ax=ax5, label=self.right_name, color='orange', alpha=0.8)
        ax5.set_title("Drawdown Comparison")
        ax5.set_ylabel("Drawdown")
        ax5.fill_between(dd_left.index, dd_left, 0, color='blue', alpha=0.2)
        ax5.fill_between(dd_right.index,
                         dd_right,
                         0,
                         color='orange',
                         alpha=0.2)
        ax5.legend()
        ax5.grid(True)

        ax6 = axes[2, 1]
        self.eval_left.factor_data['turnover'].rolling(60).mean().plot(
            ax=ax6, label=self.left_name, color='blue')
        self.eval_right.factor_data['turnover'].rolling(60).mean().plot(
            ax=ax6, label=self.right_name, color='orange')
        ax6.set_title("Turnover Comparison")
        ax6.legend()
        ax6.grid(True)

        for ax in [ax1, ax3, ax5, ax6]:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()
        self.figure = fig

    def save_results(self, base_output_dir: str):
        """
        接收一个 Figure 对象并保存所有结果。
        【修改】将 nav, ic, turnover 序列保存为独立文件。
        """
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

        #output_dir = os.path.join(base_output_dir, f"{self.left_name}_vs_{self.right_name}_{timestamp}")
        name = self.name if isinstance(self.name, str) else timestamp
        output_dir = os.path.join(base_output_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to: {output_dir}")

        plot_output_dir = os.path.join(base_output_dir, "plot")
        os.makedirs(plot_output_dir, exist_ok=True)

        # 1. 保存绩效文本 (不变)
        summary_path = os.path.join(output_dir, "performance_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(self._generate_stats_text())
        print(f"Performance summary saved to: {summary_path}")

        # --- 2. 修改：将时间序列保存为独立文件 ---
        print("Saving time series data as separate files...")

        # 定义要保存的序列和它们的来源
        series_to_save = ['nav', 'ic', 'turnover']
        instruments = {
            self.left_name: self.eval_left.factor_data,
            self.right_name: self.eval_right.factor_data
        }

        # 循环遍历每个品种和每个指标，并单独保存
        for instrument_name, data_df in instruments.items():
            for metric_name in series_to_save:
                if metric_name in data_df.columns:
                    # 构造文件名，例如: ims_nav.csv
                    file_name = f"{instrument_name}_{metric_name}.csv"
                    file_path = os.path.join(output_dir, file_name)

                    # 提取该序列并保存
                    series = data_df[metric_name]
                    series.to_csv(file_path,
                                  header=True)  # header=True 会为值列添加列名
                    print(f" -> Saved {file_path}")

        # 3. 保存传入的 Figure 对象 (不变)
        image_path = os.path.join(output_dir, "comparison_plot.png")
        self.figure.savefig(image_path, dpi=300)
        print(f"Comparison plot saved to: {image_path}")

        image_new_path = os.path.join(plot_output_dir, "{0}.png".format(name))
        self.figure.savefig(image_new_path, dpi=300)
        plt.close(self.figure)


def calc_all(expression, total_data1, period):
    total_data2 = total_data1.set_index('trade_time')
    factor_data1 = calc_expression(expression=expression,
                                   total_data=total_data2)
    dt = aggregation_data(factor_data=factor_data1,
                          returns_data=total_data1,
                          period=period)
    evaluate1 = FactorEvaluate1(factor_data=dt,
                                factor_name='transformed',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=15,
                                fee=0.000,
                                scale_method='roll_zscore',
                                name=create_id(generate_simple_id(expression)),
                                expression=expression)
    evaluate1.run()
    return evaluate1
