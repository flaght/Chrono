import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.commons.utils import JsCode  # For more advanced label formatting if needed
from ultron.kdutils.progress import Progress
from ultron.utilities.logger import kd_logger


def echarts_his_trade(orders,
                      kl_pd,
                      y_zoon=1.5,
                      time_fmt='%Y-%m-%d %H:%M:%S',
                      time_name='trade_time',
                      price_name='close'):

    # 类型检查：kl_pd 必须是 pandas DataFrame
    if not isinstance(kl_pd, pd.DataFrame):
        raise TypeError("kl_pd must be a pandas DataFrame.")
    # 检查 kl_pd 是否为空
    if kl_pd.empty:
        raise ValueError("kl_pd cannot be empty.")

    # 检查 price_name 列是否存在于 kl_pd 中
    if price_name not in kl_pd.columns:
        raise ValueError(
            f"price_name '{price_name}' not found in kl_pd columns.")

    # 检查 kl_pd 的索引是否为 DatetimeIndex
    if not isinstance(kl_pd.index, pd.DatetimeIndex):
        try:
            kl_pd = kl_pd.copy()  # 创建 kl_pd 的副本，避免修改原始数据
            kl_pd.index = pd.to_datetime(kl_pd.index)
            kd_logger.info("Converted kl_pd.index to DatetimeIndex.")
        except Exception as e:
            kd_logger.error(
                f"Failed to convert kl_pd.index to DatetimeIndex: {e}")
            # 如果转换失败，且 time_name 列存在，则尝试使用 time_name 列作为索引
            if time_name in kl_pd.columns:
                try:
                    kl_pd = kl_pd.copy()
                    kl_pd[time_name] = pd.to_datetime(kl_pd[time_name])
                    kl_pd = kl_pd.set_index(time_name)
                    kd_logger.info(
                        f"Using '{time_name}' column as DatetimeIndex.")
                except Exception as e_col:
                    kd_logger.error(
                        f"Failed to use '{time_name}' column as DatetimeIndex: {e_col}"
                    )
                    raise ValueError(
                        "kl_pd must have a DatetimeIndex or a convertible time column specified by 'time_name'."
                    ) from e_col
            else:
                raise ValueError(
                    "kl_pd.index is not a DatetimeIndex and 'time_name' column not found or invalid."
                ) from e

    # 将时间索引转换为字符串列表，用于 Pyecharts 的 X 轴
    times_str_list = [dt.strftime(time_fmt) for dt in kl_pd.index]

    # 获取价格列表
    prices_list_raw = kl_pd[price_name].tolist()

    # 将价格列表中的 NaN 值替换为 None，以便 Pyecharts 可以处理
    prices_list = [p if pd.notna(p) else None for p in prices_list_raw]

    # 检查价格列表中是否包含有效的数值数据
    if not any(p is not None for p in prices_list):
        raise ValueError(
            "Price list contains no valid numeric data after processing kl_pd."
        )

    # 获取最后一个有效的价格，用于确定当前价格
    last_valid_price_series = kl_pd[price_name].dropna().iloc[-1:]
    if not last_valid_price_series.empty:
        now_price = last_valid_price_series.iloc[0]
    else:
        kd_logger.warning(
            "No valid prices in kl_pd to determine 'now_price'. Defaulting to 0."
        )
        now_price = 0  # 如果没有有效价格，则默认当前价格为 0

    # 获取有效的价格数据，用于确定 Y 轴的范围
    valid_prices = kl_pd[price_name].dropna()
    if valid_prices.empty:
        kd_logger.warning(
            f"No valid numeric prices found in '{price_name}' column. Using default y-axis range."
        )
        min_close_overall = 0.0  # 如果没有有效价格，则使用默认 Y 轴范围
        max_close_overall = 1.0
    else:
        min_close_overall = valid_prices.min()  # 最小价格
        max_close_overall = valid_prices.max()  # 最大价格

    # 计算价格范围，用于确定 Y 轴的 padding
    range_val = max_close_overall - min_close_overall
    if pd.isna(range_val):  # 则设置 padding 为 1.0
        padding = 1.0
    elif range_val == 0:
        padding = abs(
            min_close_overall
        ) * 0.1 if min_close_overall != 0 else 1.0  # 则设置 padding 为最小价格的 10%，如果最小价格是 0，则设置为 1.0
    else:
        padding = range_val * 0.05  # 否则设置 padding 为价格范围的 5% # Y轴压缩

    if pd.isna(padding) or padding == 0:
        padding = 1.0  # 兜底处理

    # 计算 Y 轴的最小值和最大值
    ylim_min_val = min_close_overall - padding
    ylim_max_val = max_close_overall + padding

    # 设置图表的宽度和高度
    #chart_width = "1200px"
    ax_cnt = 1.5 * len(orders)
    chart_width = f"{int(1400 + ax_cnt * 10)}px"
    chart_height = f"{int(400 * y_zoon)}px"
    #pdb.set_trace()
    # 创建 Line 对象，并设置初始化参数
    line_chart = (
        Line(init_opts=opts.InitOpts(width=chart_width,
                                     height=chart_height,
                                     theme='light'))  # 设置图表宽度、高度和主题
        .add_xaxis(xaxis_data=times_str_list)  # 添加 X 轴数据
        .add_yaxis(
            series_name="Price",  # 序列名称
            y_axis=prices_list,  # 添加Y轴数据
            is_smooth=True,  # 是否平滑曲线
            linestyle_opts=opts.LineStyleOpts(width=2, color="blue"),  # 设置线条颜色
            label_opts=opts.LabelOpts(is_show=False),  # 不显示标签
            itemstyle_opts=opts.ItemStyleOpts(border_width=0,
                                              opacity=0),  # 隐藏坐标点圆点
            z_level=10  # 设置图层级别，保证在其他元素之上
        ).set_global_opts(
            title_opts=opts.TitleOpts(title=orders[0].buy_symbol
                                      if orders else "Trade History"),  # 设置标题
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

    # 用于存储标记点的配置数据
    mark_points_data = []
    # 创建时间到字符串的映射，提高效率
    time_to_str_map = {dt: dt.strftime(time_fmt) for dt in kl_pd.index}

    # 定义通用的标签格式化函数
    common_label_formatter = JsCode(
        "function(params){ return params.value.replace('\\n', '\\n'); }")
    # 定义通用的标签颜色
    common_label_color = "black"
    # 定义通用的标签字体大小
    common_label_font_size = 18

    # 循环处理每个订单，添加买入和卖出标记点
    with Progress(len(orders), 0) as pg:
        for index, order in enumerate(orders):
            pg.show(index + 1)

            buy_time_dt = pd.to_datetime(
                order.buy_time)  # 将买入时间转换为 datetime 对象
            st_key_val = -1  # 买入时间对应的索引，初始化为 -1
            buy_time_closest_dt = pd.NaT  # 最接近的买入时间，初始化为 NaT (Not a Time)

            try:
                # 查找最接近的买入时间
                buy_time_closest_dt = kl_pd.index.asof(buy_time_dt)
                # 如果找不到最接近的时间，则查找未来匹配
                if pd.isna(buy_time_closest_dt):
                    future_matches = kl_pd.index[kl_pd.index >= buy_time_dt]
                    if not future_matches.empty:
                        buy_time_closest_dt = future_matches[0]
                    else:  # 如果未来没有匹配项，则跳过
                        kd_logger.warning(
                            f"Order buy_time {buy_time_dt} not found in kl_pd range for order {index}. Skipping trade visualization parts."
                        )
                        continue

                # 获取买入时间对应的价格
                _buy_price_at_time_raw = kl_pd.loc[buy_time_closest_dt,
                                                   price_name]

                # 如果价格为 NaN，则跳过
                if pd.isna(_buy_price_at_time_raw):
                    kd_logger.warning(
                        f"Price at buy_time_closest_dt {buy_time_closest_dt} is NaN for order {index}. Skipping trade visualization."
                    )
                    continue
                buy_price_at_time = _buy_price_at_time_raw

                # 获取用于绘图的买入时间字符串
                buy_time_str_for_plot = time_to_str_map[buy_time_closest_dt]
                # 获取买入时间对应的索引
                st_key_val = kl_pd.index.get_loc(buy_time_closest_dt)

            except KeyError:
                kd_logger.error(
                    f"KeyError processing buy_time for order {index} at {buy_time_dt}. Closest kline time: {buy_time_closest_dt}. Skipping trade visualization parts."
                )
                continue
            except Exception as e_buy:
                kd_logger.error(
                    f"Generic error processing buy_time for order {index}: {e_buy}. Skipping trade visualization parts."
                )
                continue

            order_win = False  # 订单是否盈利
            effective_sell_price = None  # 有效的卖出价格
            rv_pd_end_idx_exclusive = -1  # 卖出时间对应的索引（不包含），初始化为 -1
            sell_time_dt_for_plot = None  # 用于绘图的卖出时间
            sell_price_at_time_raw = now_price  # 卖出时间对应的价格，默认为当前价格

            if order.sell_type == 'keep':
                sell_time_dt_for_plot = kl_pd.index[-1]  # 卖出时间为最后一个 K 线的时间
                _sell_price_temp = kl_pd.loc[sell_time_dt_for_plot, price_name]
                sell_price_at_time_raw = _sell_price_temp if pd.notna(
                    _sell_price_temp) else now_price  # 如果价格为 NaN，则使用当前价格
                effective_sell_price = sell_price_at_time_raw  # 有效卖出价格为卖出时间价格
                rv_pd_end_idx_exclusive = len(kl_pd)  # 卖出索引为 K 线数据的长度
                if hasattr(order, 'buy_price') and order.buy_price is not None:
                    direction_multiplier = getattr(order, 'expect_direction',
                                                   1)  # 获取预期方向
                    order_win = (effective_sell_price - order.buy_price
                                 ) * direction_multiplier > 0  # 计算是否盈利
            else:
                sell_time_dt = pd.to_datetime(order.sell_time)  # 卖出时间
                sell_time_closest_dt = pd.NaT  # 最接近的卖出时间
                try:
                    sell_time_closest_dt = kl_pd.index.asof(
                        sell_time_dt)  # 查找最接近的卖出时间
                    if pd.isna(sell_time_closest_dt):
                        future_matches = kl_pd.index[kl_pd.index >=
                                                     sell_time_dt]
                        if not future_matches.empty:
                            sell_time_closest_dt = future_matches[0]
                        else:  # 如果找不到最接近的时间，则使用最后一个数据点
                            kd_logger.warning(
                                f"Order sell_time {sell_time_dt} is after last kl_pd point for order {index}. Using last data point for plotting."
                            )
                            sell_time_closest_dt = kl_pd.index[-1]

                    sell_time_dt_for_plot = sell_time_closest_dt  # 用于绘图的卖出时间
                    _sell_price_temp = kl_pd.loc[sell_time_dt_for_plot,
                                                 price_name]
                    sell_price_at_time_raw = _sell_price_temp if pd.notna(
                        _sell_price_temp) else now_price  # 如果价格为 NaN，则使用当前价格
                    rv_pd_end_idx_exclusive = kl_pd.index.get_loc(
                        sell_time_closest_dt) + 1  # 卖出索引
                    effective_sell_price = order.sell_price if hasattr(
                        order, 'sell_price'
                    ) and order.sell_price is not None else sell_price_at_time_raw  # 有效卖出价格

                except KeyError:
                    kd_logger.error(
                        f"KeyError processing sell_time for order {index} at {sell_time_dt}. Closest kline time: {sell_time_closest_dt}. Using defaults."
                    )
                    sell_time_dt_for_plot = kl_pd.index[-1]  # 卖出时间为最后一个 K 线的时间
                    _sell_price_temp = kl_pd.loc[sell_time_dt_for_plot,
                                                 price_name]
                    sell_price_at_time_raw = _sell_price_temp if pd.notna(
                        _sell_price_temp) else now_price  # 如果价格为 NaN，则使用当前价格
                    effective_sell_price = order.sell_price if hasattr(
                        order, 'sell_price'
                    ) and order.sell_price is not None else sell_price_at_time_raw  # 有效卖出价格
                    rv_pd_end_idx_exclusive = len(kl_pd)
                except Exception as e_sell:
                    kd_logger.error(
                        f"Generic error processing sell_time for order {index}: {e_sell}. Using defaults."
                    )
                    sell_time_dt_for_plot = kl_pd.index[-1]  # 卖出时间为最后一个 K 线的时间
                    _sell_price_temp = kl_pd.loc[sell_time_dt_for_plot,
                                                 price_name]
                    sell_price_at_time_raw = _sell_price_temp if pd.notna(
                        _sell_price_temp) else now_price
                    effective_sell_price = order.sell_price if hasattr(
                        order, 'sell_price'
                    ) and order.sell_price is not None else sell_price_at_time_raw
                    rv_pd_end_idx_exclusive = len(kl_pd)

                # 根据订单类型或价格计算是否盈利
                if hasattr(order, 'sell_type') and order.sell_type == 'win':
                    order_win = True
                elif hasattr(
                        order, 'buy_price'
                ) and order.buy_price is not None and effective_sell_price is not None:
                    direction_multiplier = getattr(order, 'expect_direction',
                                                   1)
                    order_win = (effective_sell_price -
                                 order.buy_price) * direction_multiplier > 0
                else:
                    order_win = False

            sell_price_at_time_for_plot = sell_price_at_time_raw if pd.notna(
                sell_price_at_time_raw) else now_price  # 用于绘图的卖出价格

            # 添加交易区域的填充色
            if st_key_val != -1 and rv_pd_end_idx_exclusive != -1 and st_key_val < rv_pd_end_idx_exclusive:
                trade_area_prices = [None] * len(prices_list)  # 初始化交易区域价格列表
                segment_prices_raw = kl_pd[price_name].iloc[
                    st_key_val:rv_pd_end_idx_exclusive]  # 获取交易区域的价格数据
                segment_prices = [
                    p if pd.notna(p) else None
                    for p in segment_prices_raw.tolist()
                ]  # 处理 NaN 值

                if any(p is not None for p in segment_prices):
                    for i in range(len(segment_prices)):
                        trade_area_prices[st_key_val +
                                          i] = segment_prices[i]  # 填充交易区域价格列表

                    area_color = "rgba(0, 128, 0, 0.3)" if not order_win else "rgba(255, 0, 0, 0.3)"  # 根据是否盈利设置区域颜色

                    line_chart.add_yaxis(
                        series_name="",  # 序列名称为空
                        y_axis=trade_area_prices,  # 交易区域价格
                        is_smooth=False,  # 不平滑曲线
                        linestyle_opts=opts.LineStyleOpts(width=0),  # 线条宽度为 0
                        areastyle_opts={
                            "color": area_color,  # 区域颜色
                            "origin": round(ylim_min_val, 4)  # 区域起始位置
                        },
                        label_opts=opts.LabelOpts(is_show=False),  # 不显示标签
                        tooltip_opts=opts.TooltipOpts(is_show=False),  # 不显示提示框
                        z_level=1)
            else:
                kd_logger.warning(
                    f"Invalid time range for order {index} (buy_idx: {st_key_val}, sell_idx_excl: {rv_pd_end_idx_exclusive}). Skipping area fill for this trade."
                )

            buy_tip_price_actual = order.buy_price if hasattr(
                order, 'buy_price'
            ) and order.buy_price is not None else buy_price_at_time
            buy_tip = f"Buy: {buy_tip_price_actual:.2f}"  # 买入提示信息 两位小数

            buy_item_style = opts.ItemStyleOpts(color="red").opts  # 买入标记点样式
            buy_label_opts = opts.LabelOpts(  # 买入标签样式
                position='bottom',
                color=common_label_color,
                font_size=common_label_font_size,
                formatter=common_label_formatter).opts

            mark_points_data.append({  # 添加买入标记点
                "name":
                "Buy",
                "coord": [buy_time_str_for_plot, buy_price_at_time],
                "value":
                buy_tip,
                "symbol":
                'arrow',
                "symbolSize":
                18,
                "symbolRotate":
                180,
                "itemStyle":
                buy_item_style,
                "label":
                buy_label_opts
            })

            pfr = 0.0  # 盈亏比例
            sell_or_open_label = "Sell"  # 卖出或开仓标签
            sell_tip_price_display = effective_sell_price  # 卖出提示的价格

            # 如果是 "keep" 类型的订单，则显示 "Open" 标签
            if hasattr(order, 'sell_type') and order.sell_type == 'keep':
                sell_or_open_label = "Open"

            # 计算盈亏比例
            if hasattr(
                    order, 'buy_price'
            ) and order.buy_price is not None and order.buy_price != 0 and effective_sell_price is not None:
                direction_multiplier = getattr(order, 'expect_direction',
                                               1)  # 获取预期方向
                pfr = ((effective_sell_price - order.buy_price) /
                       order.buy_price) * direction_multiplier

            sell_tip = f"{sell_or_open_label}: {sell_tip_price_display:.2f}\nP/L: {pfr*100:.2f}%"  # 卖出提示信息

            if sell_time_dt_for_plot is None:
                kd_logger.warning(
                    f"sell_time_dt_for_plot is None for order {index}, cannot add sell mark point."
                )
            else:
                sell_time_str_for_plot = time_to_str_map.get(
                    sell_time_dt_for_plot)  # 获取用于绘图的卖出时间字符串
                if sell_time_str_for_plot is None:
                    kd_logger.warning(
                        f"Could not map sell_time_dt_for_plot {sell_time_dt_for_plot} to string for order {index}."
                    )
                else:
                    sell_item_style = opts.ItemStyleOpts(
                        color="green" if order_win else "purple"
                    ).opts  # 卖出标记点样式
                    sell_label_opts = opts.LabelOpts(
                        position='top',
                        color=common_label_color,
                        font_size=common_label_font_size,
                        formatter=common_label_formatter).opts

                    mark_points_data.append({  # 添加卖出标记点
                        "name":
                        sell_or_open_label,
                        "coord":
                        [sell_time_str_for_plot, sell_price_at_time_for_plot],
                        "value":
                        sell_tip,
                        "symbol":
                        'arrow',
                        "symbolSize":
                        12,
                        "itemStyle":
                        sell_item_style,
                        "label":
                        sell_label_opts
                    })

    # MODIFIED: Pass markpoint_opts as a dictionary, using 'zlevel' (ECharts native)
    line_chart.set_series_opts(markpoint_opts={
        "data": mark_points_data,
        "zlevel": 20  # ECharts native property name
    })

    return line_chart
