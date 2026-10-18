import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.commons.utils import JsCode  # For more advanced label formatting if needed
from ultron.utilities.logger import kd_logger


def echarts_his_profit(at_pd,
                       at_name,
                       y_zoon=1.5,
                       time_fmt='%Y-%m-%d %H:%M:%S',
                       file_name=None):
    # 类型检查：kl_pd 必须是 pandas DataFrame
    if not isinstance(at_pd, pd.DataFrame):
        raise TypeError("at_pd must be a pandas DataFrame.")
    # 检查 kl_pd 是否为空
    if at_pd.empty:
        raise ValueError("at_pd cannot be empty.")

    # 检查 at_name 列是否存在于 at_pd 中
    if at_name not in at_pd.columns:
        raise ValueError(f"at_name '{at_name}' not found in at_pd columns.")

    # 检查 at_pd 的索引是否为 DatetimeIndex
    if not isinstance(at_pd.index, pd.DatetimeIndex):
        try:
            at_pd = at_pd.copy()  # 创建 kl_pd 的副本，避免修改原始数据
            at_pd.index = pd.to_datetime(at_pd.index)
            kd_logger.info("Converted at_pd.index to DatetimeIndex.")
        except Exception as e:
            kd_logger.error(
                f"Failed to convert at_pd.index to DatetimeIndex: {e}")
            # 如果转换失败，且 time_name 列存在，则尝试使用 time_name 列作为索引
            if at_name in at_pd.columns:
                try:
                    at_pd = at_pd.copy()
                    at_pd.index = pd.to_datetime(at_pd.index)
                except Exception as e_col:
                    raise ValueError(
                        "at_pd must have a DatetimeIndex or a convertible time column specified by 'time_name'."
                    ) from e_col
            else:
                raise ValueError(
                    "at_pd.index is not a DatetimeIndex and 'time_name' column not found or invalid."
                ) from e

    # 将时间索引转换为字符串列表，用于 Pyecharts 的 X 轴
    times_str_list = [dt.strftime(time_fmt) for dt in at_pd.index]

    # 获取权益列表
    at_list_raw = at_pd[at_name].tolist()

    # 列表中的 NaN 值替换为 None，以便 Pyecharts 可以处理
    at_list = [p if pd.notna(p) else None for p in at_list_raw]

    # 检查价格列表中是否包含有效的数值数据
    if not any(p is not None for p in at_list):
        raise ValueError(
            "Asset list contains no valid numeric data after processing at_pd."
        )

    # 获取有效的价格数据，用于确定 Y 轴的范围
    valid_asset = at_pd[at_name].dropna()

    if valid_asset.empty:
        kd_logger.warning(
            f"No valid numeric assets found in '{at_name}' column. Using default y-axis range."
        )
        min_asset_overall = 0.0  # 如果没有有效价格，则使用默认 Y 轴范围
        max_asset_overall = 1.0
    else:
        min_asset_overall = valid_asset.min()  # 最小价格
        max_asset_overall = valid_asset.max()  # 最大价格

    # 计算价格范围，用于确定 Y 轴的 padding
    range_val = max_asset_overall - min_asset_overall
    if pd.isna(range_val):  # 则设置 padding 为 1.0
        padding = 1.0
    elif range_val == 0:
        padding = abs(
            min_asset_overall
        ) * 0.1 if min_asset_overall != 0 else 1.0  # 则设置 padding 为最小价格的 10%，如果最小价格是 0，则设置为 1.0
    else:
        padding = range_val * 0.05  # 否则设置 padding 为价格范围的 5% # Y轴压缩

    if pd.isna(padding) or padding == 0:
        padding = 1.0  # 兜底处理

    # 计算 Y 轴的最小值和最大值
    ylim_min_val = min_asset_overall - padding
    ylim_max_val = max_asset_overall + padding

    # 设置图表的宽度和高度
    chart_width = "1200px"
    chart_height = f"{int(400 * y_zoon)}px"

    # 创建 Line 对象，并设置初始化参数
    line_chart = (
        Line(init_opts=opts.InitOpts(width=chart_width,
                                     height=chart_height,
                                     theme='light'))  # 设置图表宽度、高度和主题
        .add_xaxis(xaxis_data=times_str_list)  # 添加 X 轴数据
        .add_yaxis(
            series_name=at_name,  # 序列名称
            y_axis=at_list,  # 添加Y轴数据
            is_smooth=True,  # 是否平滑曲线
            linestyle_opts=opts.LineStyleOpts(width=2, color="blue"),  # 设置线条颜色
            label_opts=opts.LabelOpts(is_show=False),  # 不显示标签
            itemstyle_opts=opts.ItemStyleOpts(border_width=0,
                                              opacity=0),  # 隐藏坐标点圆点
            z_level=10  # 设置图层级别，保证在其他元素之上
        ).set_global_opts(
            title_opts=opts.TitleOpts(title="Profit History"),  # 设置标题
            tooltip_opts=opts.TooltipOpts(trigger="axis",
                                          axis_pointer_type="cross"),  # 设置提示框
            xaxis_opts=opts.AxisOpts(
                type_="category",  # 设置 X 轴类型为 category
                axislabel_opts=opts.LabelOpts(rotate=30),  # 设置 X 轴标签旋转角度
                splitline_opts=opts.SplitLineOpts(is_show=True),  # 显示分割线
                interval=20,  # 设置 X 轴刻度间隔
            ),
            yaxis_opts=opts.AxisOpts(
                min_=round(ylim_min_val, 4),  # 设置 Y 轴最小值
                max_=round(ylim_max_val, 4),  # 设置 Y 轴最大值
                splitline_opts=opts.SplitLineOpts(is_show=True),  # 显示分割线
                axislabel_opts=opts.LabelOpts(formatter=JsCode(
                    "function (value) { return value.toFixed(4); }"))
            ),  # 设置 Y 轴标签格式化函数，保留两位小数
            legend_opts=opts.LegendOpts(pos_top="2%",
                                        is_show=True),  # 设置图例位置和显示
            datazoom_opts=[
                opts.DataZoomOpts(type_="slider",
                                  xaxis_index=0,
                                  range_start=0,
                                  range_end=100),  # 添加滑动条 DataZoom
                opts.DataZoomOpts(type_="inside",
                                  xaxis_index=0,
                                  range_start=0,
                                  range_end=100),  # 添加内部 DataZoom
            ],
        ))

    if isinstance(file_name, str):
        line_chart.render(file_name)  # 渲染图表到 HTML 文件
