import pdb, os
import numpy as np
import pandas as pd
from typing import Any
from pymongo import UpdateOne
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import matplotlib.dates as mdates
from kdutils.data import *
from config.contract import INSTRUMENTS_CODES
from lib.ret001 import create_chg, create_yields

## -------------> 数据读写


## 提取研究环境数据(通联数据)
def fetch_bench_data(instruments,
                     begin_time,
                     end_time,
                     adjusted_method='pcr',
                     forced_alignment=False):
    market_data = fetch_main_market(begin_date=begin_time,
                                    end_date=end_time,
                                    codes=[INSTRUMENTS_CODES[instruments]],
                                    method=adjusted_method,
                                    keep_symbol=True,
                                    forced_alignment=forced_alignment)
    market_data = market_data.set_index(['trade_time', 'code'])

    prev_close = market_data.groupby(level='code')['close'].shift(1)

    # np.log 在遇到 NaN 时会安全返回 NaN
    market_data['chg'] = np.log(market_data['close'] / prev_close)

    return market_data


## 提取研究环境数据（CTP聚合数据）
def fetch_research_data(instruments,
                        begin_time,
                        end_time,
                        adjusted_method='pcr',
                        **kwargs):

    # pdb.set_trace()
    market_data = fetch_local_market1(base_path=os.environ['BAR_FUT_DIRS'],
                                      begin_date=begin_time,
                                      end_date=end_time,
                                      codes=[INSTRUMENTS_CODES[instruments]],
                                      method=adjusted_method,
                                      keep_symbol=True)

    market_data = market_data.set_index(['trade_time', 'code'])

    prev_close = market_data.groupby(level='code')['close'].shift(1)

    # np.log 在遇到 NaN 时会安全返回 NaN
    market_data['chg'] = np.log(market_data['close'] / prev_close)

    return market_data


## 提取交易环境数据
def fetch_trader_data(instruments,
                      begin_time,
                      end_time,
                      adjusted_method='pcr',
                      **kwargs):

    market_data = fetch_trader_market1(begin_time=begin_time,
                                       end_time=end_time,
                                       code=INSTRUMENTS_CODES[instruments],
                                       adjusted_method=adjusted_method)

    market_data = market_data.set_index(['trade_time', 'code'])

    prev_close = market_data.groupby(level='code')['close'].shift(1)

    # np.log 在遇到 NaN 时会安全返回 NaN
    market_data['chg'] = np.log(market_data['close'] / prev_close)

    ## CTP录制需要做偏移 手动对比后
    # times = pd.to_datetime(market_data.index.get_level_values('trade_time'))
    # codes = market_data.index.get_level_values('code')
    # shifted_times = times + pd.Timedelta(minutes=1)
    # market_data.index = pd.MultiIndex.from_arrays([shifted_times, codes],
    #                                               names=['trade_time', 'code'])

    return market_data


## 提取因子绩效
def fetch_factors_metrics(mongo_client,
                          code,
                          begin_time,
                          end_time,
                          category=None,
                          names=None):
    query = {
        "code": code,
        "trade_time": {
            "$gte": begin_time,
            "$lte": end_time
        }
    }

    if category is not None:
        t_category = category if isinstance(category, list) else [category]
        query['category'] = {"$in": t_category}

    if names is not None:
        t_name = names if isinstance(names, list) else [names]
        query['name'] = {"$in": t_name}

    cursor = mongo_client[os.environ['MG_COLL']]["realm_factors_metrics"].find(
        query)
    print(query)
    results = pd.DataFrame(list(cursor))
    results = results.drop(['_id'], axis=1) if not results.empty else results
    return results


## 提取er值绩效
def fetch_netout_metrics(mongo_client,
                         code,
                         begin_time,
                         end_time,
                         category=None,
                         task_id=None):
    query = {
        "code": code,
        "task_id": task_id,
        "trade_date": {
            "$gte": begin_time,
            "$lte": end_time
        }
    }

    if category is not None:
        t_category = category if isinstance(category, list) else [category]
        query['category'] = {"$in": t_category}

    print(query)
    cursor = mongo_client[os.environ['MG_COLL']]["realm_netout_metrics"].find(
        query)

    results = pd.DataFrame(list(cursor))
    results = results.drop(['_id'], axis=1) if not results.empty else results
    return results


## 更新收益率
def update_returns_series(mongo_client, series_data, table_name, category,
                          code):
    if series_data is None or series_data.empty:
        print(f"⚠️ 表 [{table_name}] 接收到的数据为空，跳过存储。")
        return

    db = mongo_client['neutron']
    operations = []

    if pd.api.types.is_datetime64_any_dtype(series_data.index):
        formatted_times = series_data.index.strftime(
            '%Y-%m-%d %H:%M:%S').tolist()
    else:
        formatted_times = [str(t) for t in series_data.index]

    operations = []
    for t_time, val in zip(formatted_times, series_data):
        if pd.isna(val):
            continue

        filter_query = {
            'trade_time': t_time,
            'category': category,
            'code': code
        }
        # 要更新或插入的具体数据
        update_data = {
            '$set': {
                'value': float(val),  # 序列的具体数值
            }
        }
        operations.append(UpdateOne(filter_query, update_data, upsert=True))

    if operations:
        db[table_name].bulk_write(operations,
                                  ordered=False,
                                  bypass_document_validation=True)
        print(f"✅ 成功 Upsert {len(operations)} 条有效数据至表: [{table_name}]")
    else:
        print(f"⚠️ 表 [{table_name}] 没有有效数据需要存储。")


## 更新预测值
def update_netout_series1(mongo_client, series_data, table_name, category, name='value'):
    """
    将包含多列信息的 DataFrame 极速 Upsert 到 MongoDB。
    要求 df_data 必须包含: ['trade_time', 'symbol', 'value', 'code', 'task_id']
    """
    if series_data is None or series_data.empty:
        print(f"⚠️ 表 [{table_name}] 接收到的数据为空，跳过存储。")
        return

    db = mongo_client['neutron']
    operations = []

    # 1. 统一转换时间格式 (如果 trade_time 是 datetime 对象)
    # 使用 .copy() 避免 SettingWithCopyWarning
    df = series_data.copy()
    if pd.api.types.is_datetime64_any_dtype(df['trade_time']):
        df['trade_time'] = df['trade_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        df['trade_time'] = df['trade_time'].astype(str)

    # 2. 过滤掉 value 为 NaN 的行，减少数据库垃圾数据
    df = df.dropna(subset=[name])

    if df.empty:
        print(f"⚠️ 表 [{table_name}] 过滤 NaN 后没有有效数据需要存储。")
        return

    # 3. 使用 itertuples 遍历 DataFrame 构建批量操作 (itertuples 速度极快)
    for row in df.itertuples(index=False):
        # 构建唯一索引查询条件 (用于防重复插入)
        # 注意：加入了 symbol，这在期货里非常重要（防合约错乱）
        filter_query = {
            'trade_time': row.trade_time,
            'task_id': row.task_id,
            'category': category,
            'code': row.code,
            'symbol': row.symbol
        }
        # 要更新或插入的具体数据
        set1 = {
                name: float(getattr(row, name)),
                'signal': int(row.signal)
            }
        if 'value' not in set1:
            set1['value'] = row.value
        update_data = {
            '$set': set1
        }

        operations.append(UpdateOne(filter_query, update_data, upsert=True))

    # 4. 执行批量 Upsert
    if operations:
        db[table_name].bulk_write(operations,
                                  ordered=False,
                                  bypass_document_validation=True)
        print(f"✅ 成功 Upsert {len(operations)} 条有效数据至表: [{table_name}]")


## 更新绩效数据
def update_evaluate_series(mongo_client, series_data, table_name, factor_name,
                           category, code):

    db = mongo_client['neutron']
    if pd.api.types.is_datetime64_any_dtype(series_data.index):
        formatted_times = series_data.index.strftime(
            '%Y-%m-%d %H:%M:%S').tolist()
    else:
        formatted_times = [str(t) for t in series_data.index]

    operations = []
    for t_time, val in zip(formatted_times, series_data):
        if pd.isna(val):
            continue

        filter_query = {
            'trade_time': t_time,
            'name': factor_name,
            'category': category,
            'code': code
        }
        # 要更新或插入的具体数据
        update_data = {
            '$set': {
                'value': float(val),  # 序列的具体数值
            }
        }
        operations.append(UpdateOne(filter_query, update_data, upsert=True))

    if operations:
        db[table_name].bulk_write(operations,
                                  ordered=False,
                                  bypass_document_validation=True)
        print(f"✅ 成功 Upsert {len(operations)} 条有效数据至表: [{table_name}]")
    else:
        print(f"⚠️ 表 [{table_name}] 没有有效数据需要存储。")


## 更新强化学习绩效
def update_netout_series2(mongo_client, df_data, table_name, unique_keys):
    if df_data is None or df_data.empty:
        print(f"⚠️ 表 [{table_name}] 接收到的数据为空，跳过存储。")
        return

    db = mongo_client['neutron']
    collection = db[table_name]
    operations = []
    for row in df_data.itertuples(index=False):
        # 将 namedtuple 转换为字典，方便操作
        row_dict = row._asdict()

        # a. 构建唯一索引查询条件
        filter_query = {key: row_dict[key] for key in unique_keys}

        # b. 构建要更新的数据，即除了唯一键之外的所有字段
        update_data = {
            key: value
            for key, value in row_dict.items() if key not in unique_keys
        }

        # 如果 update_data 为空，可能意味着所有列都是key，跳过
        if not update_data:
            continue

        # c. 创建 UpdateOne 操作
        operations.append(
            UpdateOne(filter_query, {'$set': update_data}, upsert=True))

    # 3. 执行批量 Upsert
    if operations:
        try:
            result = collection.bulk_write(operations,
                                           ordered=False,
                                           bypass_document_validation=True)
            print(
                f"✅ 成功 Upsert {result.upserted_count + result.modified_count} 条记录至表: [{table_name}]"
            )
        except Exception as e:
            print(f"❌ 存储到 [{table_name}] 时发生错误: {e}")


## -------------> 数据计算


def create_returns(market_data, horizon, name='vwap'):
    chg_data = create_chg(market_data.reset_index(), name)
    returns_data = create_yields(data=chg_data.copy(), horizon=horizon)
    returns_data = returns_data.reset_index()
    returns_data['trade_time'] = pd.to_datetime(returns_data['trade_time'])
    returns_data = returns_data.sort_values(by=['trade_time', 'code'])
    return returns_data


def algin_data1(research_res, trader_res):
    commd_index = None
    for _, data in research_res.items():
        if commd_index is None:
            commd_index = data.index
        commd_index = commd_index.intersection(data.index)

    for k, data in trader_res.items():
        if commd_index is None:
            commd_index = data.index
        commd_index = commd_index.intersection(data.index)

    for k, data in research_res.items():
        research_res[k] = data.loc[commd_index]

    for k, data in trader_res.items():
        trader_res[k] = data.loc[commd_index]

    return research_res, trader_res


def algin_data2(research_data, trader_data):
    commd_index = research_data.index.intersection(trader_data.index)
    research_data = research_data.loc[commd_index]
    trader_data = trader_data.loc[commd_index]
    assert research_data.index.equals(trader_data.index)
    return research_data, trader_data


def market_data_format(market_data,
                       cols=[
                           'close', 'high', 'low', 'open', 'value', 'volume',
                           'openint', 'chg', 'vwap'
                       ]):
    res = {}
    for col in cols:
        res[col] = market_data[col].unstack()
    return res


### 过滤非交易断时间
def filter_trading_time(
    data: Any,
    trading_sessions,
    drop_non_zero_second: bool = True,
) -> pd.DataFrame:
    prepared = data.reset_index()
    hhmm = prepared["trade_time"].dt.strftime("%H:%M")

    if not trading_sessions:
        filtered = prepared.copy()
    else:
        session_mask = pd.Series(False, index=prepared.index)
        for start_text, end_text in trading_sessions:
            session_mask |= (hhmm >= start_text) & (hhmm <= end_text)
        filtered = prepared.loc[session_mask].copy()

    if drop_non_zero_second:
        filtered = filtered.loc[filtered["trade_time"].dt.second.eq(0)].copy()

    return filtered.sort_values("trade_time").reset_index(drop=True)


def price_diff_metrics(research_market, trader_market, tick_size,
                       price_fields):
    res = []
    for col in price_fields:
        diff = research_market[col] - trader_market[col]  #价格差异
        abs_diff = abs(diff)  # 价格绝对差异
        abs_diff_tick = abs_diff / tick_size  # tick 差异 = 价格绝对差异 / 最小变动价位

        exact_match_ratio = np.mean(abs_diff_tick == 0)  # 完全一致的 bar 占比
        within_1tick_ratio = np.mean(abs_diff_tick
                                     <= 1)  # 差异不超过 1 tick 的 bar 占比
        within_2tick_ratio = np.mean(abs_diff_tick
                                     <= 2)  # 差异不超过 2 tick 的 bar 占比
        mean_abs_diff_tick = np.mean(abs_diff_tick)  # 平均 tick 差异
        median_abs_diff_tick = np.median(abs_diff_tick)  # 中位数 tick 差异
        p95_abs_diff_tick = np.quantile(abs_diff_tick,
                                        0.95)  # 95% 的 bar 差异不超过多少 tick
        p99_abs_diff_tick = np.quantile(abs_diff_tick,
                                        0.99)  # 99% 的 bar 差异不超过多少 tick
        max_abs_diff_tick = np.max(abs_diff_tick)  # 最大 tick 差异

        res.append({
            'name': col,
            'exact_match_ratio': exact_match_ratio,
            'within_1tick_ratio': within_1tick_ratio,
            'within_2tick_ratio': within_2tick_ratio,
            'mean_abs_diff_tick': mean_abs_diff_tick,
            'median_abs_diff_tick': median_abs_diff_tick,
            'p95_abs_diff_tick': p95_abs_diff_tick,
            'p99_abs_diff_tick': p99_abs_diff_tick,
            'max_abs_diff_tick': max_abs_diff_tick
        })
    return res


def relative_diff_metrics(research_market, trader_market, rel_fields):
    res = []
    for col in rel_fields:
        diff = research_market[col] - trader_market[col]
        abs_diff = abs(diff)

        denom = np.maximum(research_market[col].abs(),
                           trader_market[col].abs())

        denom = np.maximum(denom, 1)

        rel_diff = abs_diff / denom

        exact_match_ratio = np.mean(abs_diff == 0)  # 成交量完全一致比例
        mean_abs_diff = np.mean(abs_diff)  # 平均差多少手
        median_abs_diff = np.median(abs_diff)
        p95_abs_diff = np.quantile(abs_diff, 0.95)
        p99_abs_diff = np.quantile(abs_diff, 0.99)
        max_abs_diff = np.max(abs_diff)
        mean_rel_diff = np.mean(rel_diff)
        median_rel_diff = np.median(rel_diff)
        p95_rel_diff = np.quantile(rel_diff, 0.95)
        p99_rel_diff = np.quantile(rel_diff, 0.99)
        max_rel_diff = np.max(rel_diff)
        large_diff_1pct_ratio = np.mean(rel_diff > 0.01)
        large_diff_2pct_ratio = np.mean(rel_diff > 0.02)
        large_diff_5pct_ratio = np.mean(rel_diff
                                        > 0.05)  # 成交量相对误差超过 5% 的 bar 占比

        res.append({
            'name': col,
            'exact_match_ratio': exact_match_ratio,
            'mean_abs_diff': mean_abs_diff,
            'median_abs_diff': median_abs_diff,
            'p95_abs_diff': p95_abs_diff,
            'p99_abs_diff': p99_abs_diff,
            'max_abs_diff': max_abs_diff,
            'mean_rel_diff': mean_rel_diff,
            'median_rel_diff': median_rel_diff,
            'p95_rel_diff': p95_rel_diff,
            'p99_rel_diff': p99_rel_diff,
            'max_rel_diff': max_rel_diff,
            'large_diff_1pct_ratio': large_diff_1pct_ratio,
            'large_diff_2pct_ratio': large_diff_2pct_ratio,
            'large_diff_5pct_ratio': large_diff_5pct_ratio
        })
    return res


def factor_metrics(factor_research,
                   factor_trade,
                   name,
                   eps: float = 1e-12,
                   upper: float = None,
                   lower: float = None,
                   threshold: float = None):
    diff = factor_research - factor_trade
    abs_diff = diff.abs()
    denom = pd.concat([
        factor_research.abs(),
        factor_trade.abs(),
        pd.Series(eps, index=factor_research.index)
    ],
                      axis=1).max(axis=1)
    rel_diff = abs_diff / denom
    pearson_corr = factor_research.corr(factor_trade, method="pearson")
    spearman_corr = factor_research.corr(factor_trade, method="spearman")
    sign_match = np.sign(factor_research) == np.sign(factor_trade)
    zero_cross = (factor_research * factor_trade) < 0

    res = {
        "valid_count": len(factor_research),
        "mean_abs_diff": abs_diff.mean(),
        "median_abs_diff": abs_diff.median(),
        "p95_abs_diff": abs_diff.quantile(0.95),
        "p99_abs_diff": abs_diff.quantile(0.99),
        "max_abs_diff": abs_diff.max(),
        "mean_rel_diff": rel_diff.mean(),
        "median_rel_diff": rel_diff.median(),
        "p95_rel_diff": rel_diff.quantile(0.95),
        "p99_rel_diff": rel_diff.quantile(0.99),
        "max_rel_diff": rel_diff.max(),
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "sign_match_ratio": sign_match.mean(),
        "zero_cross_ratio": zero_cross.mean(),
    }

    if threshold is not None:
        signal_r = factor_research > threshold
        signal_t = factor_trade > threshold

        res["signal_match_ratio"] = (signal_r == signal_t).mean()
        res["signal_flip_ratio"] = (signal_r != signal_t).mean()

    if upper is not None and lower is not None:
        signal_r = pd.Series(0, index=factor_research.index)
        signal_t = pd.Series(0, index=factor_trade.index)

        signal_r[factor_research > upper] = 1
        signal_r[factor_research < lower] = -1

        signal_t[factor_trade > upper] = 1
        signal_t[factor_trade < lower] = -1

        res["signal_match_ratio"] = (signal_r == signal_t).mean()
        res["signal_flip_ratio"] = (signal_r != signal_t).mean()
        res["long_short_reverse_ratio"] = ((signal_r * signal_t) == -1).mean()

    res['name'] = name
    return res


## -------------> 数据绘图
# def plot_netout(data,
#                 figsize=(15, 12),
#                 marker='o',
#                 markersize=4,
#                 grid=True,
#                 bar_metrics=None,
#                 date_format='%Y-%m-%d',
#                 rotation=45,
#                 n_ticks=None,
#                 nav_as_cumret=True,
#                 ic_as_cumsum=True):

#     if bar_metrics is None:
#         bar_metrics = ['maxdd']

#     metric_groups = {
#         'Cumulative NAV Return': ['net_nav', 'gross_nav'],
#         'Cumulative IC': ['ic'],
#         'Max Drawdown': ['maxdd']
#     }

#     plot_data = data.copy()
#     plot_data['trade_date'] = pd.to_datetime(plot_data['trade_date'])
#     plot_data = plot_data.sort_values(['category', 'trade_date'])

#     # =========================
#     # 计算累计收益率 / 累计 IC
#     # =========================
#     if nav_as_cumret:
#         for col in ['net_nav', 'gross_nav']:
#             plot_data[col] = (
#                 plot_data
#                 .groupby('category')[col]
#                 .transform(lambda x: (1 + x).cumprod() - 1)
#             )

#     if ic_as_cumsum:
#         plot_data['ic'] = (
#             plot_data
#             .groupby('category')['ic']
#             .transform(lambda x: x.cumsum())
#         )

#     # 仅用于展示：NAV 放大为百分比
#     plot_data[['net_nav', 'gross_nav']] = (
#         plot_data[['net_nav', 'gross_nav']]
#     )

#     # IC 可选放大，默认不放大
#     plot_data['ic'] = plot_data['ic']

#     plot_data = plot_data.sort_values(['trade_date', 'category'])

#     n_plots = len(metric_groups)

#     fig, axes = plt.subplots(
#         n_plots,
#         1,
#         figsize=figsize,
#         sharex=True
#     )

#     if n_plots == 1:
#         axes = [axes]

#     categories = list(plot_data['category'].dropna().unique())
#     dates = sorted(plot_data['trade_date'].dropna().unique())
#     x = np.arange(len(dates))

#     color_idx = 0
#     color_map = {}

#     for metric_list in metric_groups.values():
#         for metric in metric_list:
#             for category in categories:
#                 color_map[(category, metric)] = f'C{color_idx}'
#                 color_idx += 1

#     line_style_map = {
#         'net_nav': '-',
#         'gross_nav': '--',
#         'ic': '-'
#     }

#     marker_map = {
#         'net_nav': 'o',
#         'gross_nav': 's',
#         'ic': 'o'
#     }

#     linewidth_map = {
#         'net_nav': 2.0,
#         'gross_nav': 2.8,
#         'ic': 2.0
#     }

#     alpha_map = {
#         'net_nav': 0.85,
#         'gross_nav': 1.0,
#         'ic': 0.9
#     }

#     for ax, (title, metrics) in zip(axes, metric_groups.items()):

#         for metric in metrics:

#             if metric in bar_metrics:
#                 width = 0.8 / len(categories)

#                 for i, category in enumerate(categories):
#                     tmp = (
#                         plot_data[plot_data['category'] == category]
#                         .set_index('trade_date')
#                         .reindex(dates)
#                     )

#                     offset = (i - (len(categories) - 1) / 2) * width

#                     ax.bar(
#                         x + offset,
#                         tmp[metric].values,
#                         width=width,
#                         color=color_map[(category, metric)],
#                         alpha=0.8,
#                         label=f'{category} - {metric}'
#                     )

#             else:
#                 for category in categories:
#                     tmp = (
#                         plot_data[plot_data['category'] == category]
#                         .set_index('trade_date')
#                         .reindex(dates)
#                     )

#                     ax.plot(
#                         x,
#                         tmp[metric].values,
#                         linestyle=line_style_map.get(metric, '-'),
#                         marker=marker_map.get(metric, marker),
#                         markersize=markersize,
#                         linewidth=linewidth_map.get(metric, 2.0),
#                         color=color_map[(category, metric)],
#                         alpha=alpha_map.get(metric, 0.9),
#                         label=f'{category} - {metric}',
#                         zorder=3 if metric == 'gross_nav' else 2
#                     )

#         ax.axhline(0, linestyle='--', linewidth=1, alpha=0.5)
#         ax.set_title(title)

#         if title == 'Cumulative NAV Return':
#             ax.set_ylabel('Cumulative Return x ')
#         elif title == 'Cumulative IC':
#             ax.set_ylabel('Cumulative IC')
#         else:
#             ax.set_ylabel(title)

#         ax.grid(grid)
#         ax.legend()

#     # =========================
#     # 显式设置横轴日期
#     # =========================
#     if len(dates) > 0:
#         if n_ticks is None:
#             tick_idx = np.arange(len(dates))
#         else:
#             n_ticks = min(n_ticks, len(dates))
#             tick_idx = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)

#         date_labels = [
#             pd.Timestamp(dates[i]).strftime(date_format)
#             for i in tick_idx
#         ]

#         axes[-1].set_xticks(tick_idx)
#         axes[-1].set_xticklabels(
#             date_labels,
#             rotation=rotation,
#             ha='right'
#         )

#     axes[-1].set_xlabel('trade_date')

#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.12)
#     plt.show()

#     return fig, axes


def plot_timesies(data,
                  values,
                  index='trade_time',
                  columns='category',
                  cumsum=True,
                  n_ticks=10,
                  figsize=(15, 7),
                  marker='o',
                  markersize=3,
                  grid=True,
                  mark_day_change=True,
                  title=None,
                  ylabel=None):
    df = (data.pivot_table(index=index, columns=columns,
                           values=values).sort_index())

    plot_df = df.cumsum() if cumsum else df.copy()

    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=figsize)

    for col in plot_df.columns:
        ax.plot(x,
                plot_df[col],
                marker=marker,
                markersize=markersize,
                label=col)

    if mark_day_change:
        dates = plot_df.index.date
        day_change_idx = np.where(dates[1:] != dates[:-1])[0] + 1

        for idx in day_change_idx:
            ax.axvline(idx, linestyle='--', alpha=0.3)

    if len(plot_df) > 0:
        n_ticks = min(n_ticks, len(plot_df))
        tick_idx = np.linspace(0, len(plot_df) - 1, n_ticks, dtype=int)

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(plot_df.index[tick_idx].strftime('%Y-%m-%d %H:%M'),
                           rotation=45,
                           ha='right')

    ax.grid(grid)
    ax.legend(title=columns)
    ax.set_xlabel(index)

    if ylabel is None:
        ylabel = f'cumsum {values}' if cumsum else values
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"{'Cumulative ' if cumsum else ''}{values} by Trading Bar"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    return fig, ax, plot_df


def plot_netout(data,
                figsize=(15, 12),
                marker='o',
                markersize=4,
                grid=True,
                bar_metrics=None,
                date_format='%Y-%m-%d',
                rotation=45,
                n_ticks=None,
                nav_as_cumret=True,
                ic_as_cumsum=True):

    if bar_metrics is None:
        bar_metrics = ['maxdd']

    metric_groups = {
        'Cumulative NAV Return': ['net_nav', 'gross_nav'],
        'Cumulative IC': ['ic'],
        'Profit Ratio': ['profit_ratio'],
        'Max Drawdown': ['maxdd']
    }

    plot_data = data.copy()
    plot_data['trade_date'] = pd.to_datetime(plot_data['trade_date'])
    plot_data = plot_data.sort_values(['category', 'trade_date'])

    # =========================
    # 计算累计收益率 / 累计 IC
    # =========================
    if nav_as_cumret:
        for col in ['net_nav', 'gross_nav']:
            plot_data[col] = (plot_data.groupby('category')[col].transform(
                lambda x: (1 + x).cumprod() - 1))

    if ic_as_cumsum:
        plot_data['ic'] = (plot_data.groupby('category')['ic'].transform(
            lambda x: x.cumsum()))

    # 仅用于展示：NAV 放大为百分比
    plot_data[['net_nav', 'gross_nav']] = (plot_data[['net_nav', 'gross_nav']])

    # IC 可选放大，默认不放大
    plot_data['ic'] = plot_data['ic']

    plot_data = plot_data.sort_values(['trade_date', 'category'])

    n_plots = len(metric_groups)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=False)

    if n_plots == 1:
        axes = [axes]

    categories = list(plot_data['category'].dropna().unique())
    dates = sorted(plot_data['trade_date'].dropna().unique())
    date_index = pd.DatetimeIndex(dates)
    x = mdates.date2num(date_index.to_pydatetime())

    color_idx = 0
    color_map = {}

    for metric_list in metric_groups.values():
        for metric in metric_list:
            for category in categories:
                color_map[(category, metric)] = f'C{color_idx}'
                color_idx += 1

    line_style_map = {
        'net_nav': '-',
        'gross_nav': '--',
        'ic': '-',
        'profit_ratio': '-.'
    }

    marker_map = {
        'net_nav': 'o',
        'gross_nav': 's',
        'ic': 'o',
        'profit_ratio': 'D'
    }

    linewidth_map = {
        'net_nav': 2.0,
        'gross_nav': 2.8,
        'ic': 2.0,
        'profit_ratio': 2.0
    }

    alpha_map = {
        'net_nav': 0.85,
        'gross_nav': 1.0,
        'ic': 0.9,
        'profit_ratio': 0.9
    }

    for ax, (title, metrics) in zip(axes, metric_groups.items()):

        for metric in metrics:

            if metric in bar_metrics:
                width = 0.8 / len(categories)

                for i, category in enumerate(categories):
                    tmp = (plot_data[plot_data['category'] == category].
                           set_index('trade_date').reindex(dates))

                    offset = (i - (len(categories) - 1) / 2) * width

                    ax.bar(x + offset,
                           tmp[metric].values,
                           width=width,
                           color=color_map[(category, metric)],
                           alpha=0.8,
                           label=f'{category} - {metric}')

            else:
                for category in categories:
                    tmp = (plot_data[plot_data['category'] == category].
                           set_index('trade_date').reindex(dates))

                    ax.plot(x,
                            tmp[metric].values,
                            linestyle=line_style_map.get(metric, '-'),
                            marker=marker_map.get(metric, marker),
                            markersize=markersize,
                            linewidth=linewidth_map.get(metric, 2.0),
                            color=color_map[(category, metric)],
                            alpha=alpha_map.get(metric, 0.9),
                            label=f'{category} - {metric}',
                            zorder=3 if metric == 'gross_nav' else 2)

        ax.axhline(0, linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(title)
        ax.xaxis_date()

        if title == 'Cumulative NAV Return':
            ax.set_ylabel('Cumulative Return x ')
        elif title == 'Cumulative IC':
            ax.set_ylabel('Cumulative IC')
        elif title == 'Profit Ratio':
            ax.set_ylabel('Profit Ratio')
        else:
            ax.set_ylabel(title)

        ax.grid(grid)
        ax.legend()

    # =========================
    # 显式设置横轴日期
    # =========================
    if len(dates) > 0:
        if n_ticks is None or n_ticks >= len(dates):
            tick_dates = x
        else:
            n_ticks = min(n_ticks, len(dates))
            tick_idx = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
            tick_dates = x[tick_idx]

        for ax in axes:
            ax.xaxis.set_major_locator(FixedLocator(tick_dates))

    formatter = mdates.DateFormatter(date_format)
    for ax in axes:
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', labelbottom=True)

    for ax in axes:
        ax.set_xlabel('trade_date')

    plt.tight_layout()
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
            label.set_ha('right')
    plt.show()

    return fig, axes
