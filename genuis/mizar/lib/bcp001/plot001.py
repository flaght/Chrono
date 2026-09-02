import os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
from dataclasses import dataclass

from kdutils.macro2 import base_path

from lib.rl012.sandbox import empyrical_metrics


@dataclass
class Params:
    composite_method: str
    composite_id: str
    singal_method: str
    signal_id: str
    model_id: str
    backtest_id: str


def plot_compare(left_file, right_file, left_name, right_name):
    _, axes = plt.subplots(1, 2, figsize=(20, 10))
    left_img = Image.open(left_file)
    right_img = Image.open(right_file)
    axes[0].imshow(left_img)
    axes[0].set_title(left_name)  # 设置标题
    axes[0].axis('off')  # 隐藏坐标轴（可选）

    axes[1].imshow(right_img)
    axes[1].set_title(right_name)
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def plot_together(left_path, right_path, left_name, right_name):
    left_daily = pd.read_feather(left_path)
    right_daily = pd.read_feather(right_path)
    left_metrics, left_cum_return = empyrical_metrics(left_daily)
    left_metrics['name'] = left_name
    left_cum_return.name = left_name

    right_metrics, right_cum_return = empyrical_metrics(right_daily)
    right_metrics['name'] = right_name
    right_cum_return.name = right_name

    metrics_df = pd.DataFrame([left_metrics, right_metrics])
    metrics_df.set_index('name', inplace=True)
    metrics_df = metrics_df.round(4)

    cum_pnl_left = left_daily['daily_pnl'].cumsum()
    cum_pnl_right = right_daily['daily_pnl'].cumsum()
    display(metrics_df)
    # 开始画图
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    # --- 第一张图 (放在 axes[0] 位置)：原始 Cum PnL ---
    axes[0].plot(cum_pnl_left,
                 label=f'Strategy {left_name} (Cum PnL)',
                 linewidth=2)
    axes[0].plot(cum_pnl_right,
                 label=f'Strategy {right_name} (Cum PnL)',
                 linewidth=2)

    axes[0].set_title('Cumulative PnL Comparison', fontsize=15)
    axes[0].set_ylabel('Cumulative PnL', fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # --- 第二张图 (放在 axes[1] 位置)：Empyrical Cum Return ---
    axes[1].plot(left_cum_return,
                 label=f'Strategy {left_name} (Cum Return)',
                 linewidth=2)
    axes[1].plot(right_cum_return,
                 label=f'Strategy {right_name} (Cum Return)',
                 linewidth=2)

    axes[1].set_title('Empyrical Cumulative Return Comparison', fontsize=15)
    axes[1].set_xlabel('Date / Time', fontsize=12)
    axes[1].set_ylabel('Cumulative Return', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # 展示图表
    plt.tight_layout()
    plt.show()


### er  原始方法和WF方法   绩效对比
def plot_algo_er_metrics1(method, instruments, task_id, period, model_id,
                          composite_method, composite_id, category):
    mapping = {'rl': 'model', 'equal_weight': 'linear'}
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'composite')

    proto_er = os.path.join(base_path1, mapping[composite_method],
                            composite_method, composite_id, "metrics", "or",
                            "plot", "{0}.png".format(category))

    final_er = os.path.join(base_path1, mapping[composite_method],
                            composite_method, composite_id, "metrics", "wf",
                            "plot", "{0}.png".format(category))

    return plot_compare(
        left_file=proto_er,
        right_file=final_er,
        left_name='prot_{}_{}_{}'.format(category, composite_method,
                                         composite_id),
        right_name='final_{}_{}_{}'.format(category, composite_method,
                                           model_id))


### 信号  原始方法和WF方法  绩效对比
def plot_algo_signal_metrics1(method, instruments, task_id, period,
                              composite_method, composite_id, singal_method,
                              signal_id, model_id, category):
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'signal')

    proto_signal = os.path.join(base_path1, "proto", composite_method,
                                composite_id, singal_method, "metrics", "plot",
                                "{0}_{1}.png".format(signal_id, category))

    final_signal = os.path.join(base_path1, "final", composite_method,
                                model_id, "metrics", "plot",
                                "{0}.png".format(category))

    return plot_compare(
        left_file=proto_signal,
        right_file=final_signal,
        left_name='prot_{}_{}_{}'.format(category, composite_method,
                                         composite_id),
        right_name='final_{}_{}_{}'.format(category, composite_method,
                                           model_id))


def plot_algo_backtest_metrics1(method, instruments, task_id, period,
                                composite_method, composite_id, singal_method,
                                signal_id, model_id, category, backtest_id):
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'backtest')

    proto_path = os.path.join(base_path1, "proto", composite_method,
                              backtest_id, composite_id, singal_method,
                              "{0}_{1}".format(signal_id, category),
                              "daily_stats.feather")

    final_path = os.path.join(base_path1, "final", composite_method, model_id,
                              category, "daily_stats.feather")

    plot_together(left_path=proto_path,
                  right_path=final_path,
                  left_name="proto_backtest_{0}".format(category),
                  right_name="final_backtest_{0}".format(category))


### er WF 方法对比


def _plot_mode_er_metrics1(method, instruments, task_id, period,
                           composite_method, composite_id, category):
    mapping = {'rl': 'model', 'equal_weight': 'linear'}
    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'composite')
    final_er = os.path.join(base_path1, mapping[composite_method],
                            composite_method, composite_id, "metrics", "wf",
                            "plot", "{0}.png".format(category))
    return final_er


def _plot_mode_signal_metrics1(method, instruments, task_id, period,
                               composite_method, model_id, category):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'signal')

    final_signal = os.path.join(base_path1, "final", composite_method,
                                model_id, "metrics", "plot",
                                "{0}.png".format(category))
    return final_signal


def _plot_mode_backtest_metrics(method, instruments, task_id, period,
                                composite_method, model_id, category):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'backtest')
    final_path = os.path.join(base_path1, "final", composite_method, model_id,
                              category, "daily_stats.feather")
    return final_path


def plot_mode_er_metrics1(method, instruments, task_id, period, category,
                          left_params, right_params):
    left_er = _plot_mode_er_metrics1(
        method=method,
        instruments=instruments,
        task_id=task_id,
        period=period,
        category=category,
        composite_method=left_params.composite_method,
        composite_id=left_params.composite_id)

    right_er = _plot_mode_er_metrics1(
        method=method,
        instruments=instruments,
        task_id=task_id,
        period=period,
        category=category,
        composite_method=right_params.composite_method,
        composite_id=right_params.composite_id)

    return plot_compare(
        left_file=left_er,
        right_file=right_er,
        left_name='{}_{}_{}'.format(category, left_params.composite_method,
                                    left_params.model_id),
        right_name='{}_{}_{}'.format(category, right_params.composite_method,
                                     right_params.model_id))


def plot_mode_signal_metrics1(method, instruments, task_id, period, category,
                              left_params, right_params):

    left_signal = _plot_mode_signal_metrics1(
        method,
        instruments,
        task_id,
        period,
        composite_method=left_params.composite_method,
        model_id=left_params.model_id,
        category=category)

    right_signal = _plot_mode_signal_metrics1(
        method,
        instruments,
        task_id,
        period,
        composite_method=right_params.composite_method,
        model_id=right_params.model_id,
        category=category)

    return plot_compare(
        left_file=left_signal,
        right_file=right_signal,
        left_name='{}_{}_{}'.format(category, left_params.composite_method,
                                    left_params.model_id),
        right_name='{}_{}_{}'.format(category, right_params.composite_method,
                                     right_params.model_id))


def plot_mode_backtest_metrics1(method, instruments, task_id, period, category,
                                left_params, right_params):
    left_backtest = _plot_mode_backtest_metrics(
        method=method,
        instruments=instruments,
        task_id=task_id,
        period=period,
        composite_method=left_params.composite_method,
        model_id=left_params.model_id,
        category=category)

    right_backtest = _plot_mode_backtest_metrics(
        method=method,
        instruments=instruments,
        task_id=task_id,
        period=period,
        composite_method=right_params.composite_method,
        model_id=right_params.model_id,
        category=category)

    plot_together(left_path=left_backtest,
                  right_path=right_backtest,
                  left_name="{}_{}_{}".format(category,
                                              left_params.composite_method,
                                              left_params.model_id),
                  right_name="{}_{}_{}".format(category,
                                               right_params.composite_method,
                                               right_params.model_id))
