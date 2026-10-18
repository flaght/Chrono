from ultron.utilities.logger import kd_logger
from ultron.kdutils.date import str_to_datetime
import pandas as pd
import matplotlib.pyplot as plt
import pdb


def plot_his_profit(at_pd,
                    at_name,
                    y_zoon=1.5,
                    step=100,
                    time_fmt='%Y-%m-%d %H:%M:%S',
                    file_name=None):
    all_pd = at_pd

    fig, ax = plt.subplots(figsize=(14, 8 * y_zoon))
    times = [time.strftime(time_fmt) for time in all_pd.index]

    ## 压缩Y轴
    min_close = all_pd[at_name].min()
    max_close = all_pd[at_name].max()
    padding = (max_close - min_close) * 0.1

    # Add slightly more padding specifically for annotations if needed
    annotation_padding_factor = 0.05  # Add 5% extra range for annotations
    ylim_min = min_close - padding - (max_close -
                                      min_close) * annotation_padding_factor
    ylim_max = max_close + padding + (max_close -
                                      min_close) * annotation_padding_factor
    y_range = ylim_max - ylim_min  # Calculate the full visible y-range

    ax.set_ylim(ylim_min, ylim_max)

    ### 绘制当前价格线
    ax.plot(times, all_pd[at_name])
    # 填充透明blue, 针对用户一些版本兼容问题进行处理
    ax.fill_between(times, 0, all_pd[at_name], color='blue', alpha=.18)

    ### X轴时间格式化
    #step = 100
    ax.set_xticks(range(0, len(times), step))
    ax.set_xticklabels(times[::step], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    plt.title("Profit History")
    if isinstance(file_name, str):
        plt.savefig(file_name)
