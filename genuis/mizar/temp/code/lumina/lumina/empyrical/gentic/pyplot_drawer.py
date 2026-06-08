from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

def plot_his_profit(cumulative_returns, time_periods, name):
    """
    动态绘制累计收益曲线，函数会自动适应传入的DataFrame的列数和列名。

    :param cumulative_returns: pandas.DataFrame, index是日期，columns是各个策略的收益曲线。
    :param time_periods: dict, 包含 'train_time', 'val_time', 'test_time' 的元组。
    :param name: str, 图表标题的一部分。
    """
    
    # --- 1. [核心修正：完全动态化] ---
    # 动态获取策略列表，不再写死！
    strategy_order = cumulative_returns.columns.tolist()
    num_strategies = len(strategy_order)
    
    # 动态生成颜色映射
    # 每个策略在三个时期（Train/Val/Test）的颜色都来自同一个“色系”，但深浅不同
    train_cmap = plt.cm.get_cmap('Blues')
    val_cmap = plt.cm.get_cmap('Greens')
    test_cmap = plt.cm.get_cmap('Oranges')
    
    # 为不同策略生成不同的颜色“浓度”
    color_shades = np.linspace(0.5, 1.0, num_strategies)
    
    strategy_colors = {}
    for i, strategy_name in enumerate(strategy_order):
        shade = color_shades[i]
        strategy_colors[strategy_name] = (train_cmap(shade), val_cmap(shade), test_cmap(shade))

    # 定义固定的时期顺序
    period_order = ['train_time', 'val_time', 'test_time']
    period_label_map = {'train_time': 'Train', 'val_time': 'Validation', 'test_time': 'Test'}
    
    # --- 2. 绘图部分 ---
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # [修正] 恢复原始、清晰的背景色
    ax.axvspan(time_periods['train_time'][0], time_periods['train_time'][1], color='g', alpha=0.15, zorder=0)
    ax.axvspan(time_periods['val_time'][0], time_periods['val_time'][1], color='royalblue', alpha=0.15, zorder=0)
    ax.axvspan(time_periods['test_time'][0], time_periods['test_time'][1], color='indianred', alpha=0.25, zorder=0)
    
    # 动态绘制曲线
    for i, period_name in enumerate(period_order):
        start, end = time_periods[period_name]
        period_data = cumulative_returns.loc[start:end]
        if period_data.empty: continue
        
        # 这里的循环现在是完全动态的
        for strategy_name in strategy_order:
            if strategy_name in period_data.columns:
                 ax.plot(period_data.index, period_data[strategy_name],
                         color=strategy_colors[strategy_name][i],
                         linewidth=1.5, zorder=2)

    # --- 3. [核心修正：动态构建图例] ---
    
    # 准备图例“零件”
    bg_handles = [
        Patch(facecolor='g', alpha=0.15, label='Train Period'),
        Patch(facecolor='royalblue', alpha=0.15, label='Validation Period'),
        Patch(facecolor='indianred', alpha=0.25, label='Test Period')
    ]
    
    line_handles = {p: {} for p in period_order}
    for i, p_name in enumerate(period_order):
        # 这里的循环现在也是完全动态的
        for s_name in strategy_order:
            line_handles[p_name][s_name] = Line2D([0], [0], color=strategy_colors[s_name][i], lw=2,
                                                  label=f'{s_name} ({period_label_map[p_name]})')
    
    # 按行、按列，动态地、一个一个地把“零件”放进最终列表
    final_handles = []
    for i in range(len(period_order)): # 遍历行 (Train, Val, Test)
        # 添加第一列的元素 (背景色块)
        final_handles.append(bg_handles[i])
        
    # 动态添加该行后续列的元素 (策略曲线)
    for s_name in strategy_order:
        for i in range(len(period_order)):
            final_handles.append(line_handles[period_order[i]][s_name])
            
    # --- 4. 显示图例和美化 ---
    ax.set_title(f'Cumulative Returns ({name})', fontsize=16)
    ax.set_xlabel('Trade Time', fontsize=12)
    ax.set_ylabel('Cumulative Returns', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    # 列数也是动态计算的
    num_legend_cols = 1 + num_strategies
    
    ax.legend(handles=final_handles,
              loc='upper left',  # 固定在左上角
              fontsize=9,
              ncol=num_legend_cols)

    # [修正] 设置x轴范围，消除两边空白
    ax.set_xlim(time_periods['train_time'][0], time_periods['test_time'][1])
    
    # 日期格式化
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    date_format = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()