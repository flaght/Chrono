import itertools, datetime, pdb, os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from lumina.formual.impulse import Impulse
from lumina.formual.iactuator import Iactuator

from ultron.sentry.api import *
from alphacopilot.api.calendars import advanceDateByCalendar
from kdutils.data import fetch_main_market, fetch_trader_market1
from kdutils.macro2 import base_path
from config.contract import INSTRUMENTS_CODES
from kdutils.ttimes import get_dates
from kdutils.common import fetch_temp_data


def load_data_from_feather(instruments, method, rootid=0):

    total_factors = fetch_temp_data(method=method,
                                    task_id=rootid,
                                    instruments=instruments,
                                    datasets=['train', 'val'])
    total_factors = total_factors.set_index(['trade_time', 'code'])

    res = {}
    cols = [
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'chg',
        'price', 'vwap'
    ]
    for col in cols:
        if col in total_factors.columns:
            res[col] = total_factors[col].unstack()
    return res


def load_data_from_dolphin(instruments, start_date, end_date):

    market_data = fetch_main_market(begin_date=start_date,
                                    end_date=end_date,
                                    codes=[INSTRUMENTS_CODES[instruments]],
                                    method='pcr',
                                    keep_symbol=True)
    market_data = market_data.set_index(['trade_time', 'code'])

    res = {}
    cols = [
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'chg',
        'price', 'vwap'
    ]
    for col in cols:
        res[col] = market_data[col].unstack()
    return res


### 不同因子计算方式对比 #### 同样数据源，不同计算入口
def impulse_factors1(instruments, days):
    factors_list = [
        "MADecay(30,'oi039_1_2_1')",
        "ADDED(EMA(15,'oi004_5_10_1'),MADiff(240,'oi004_1_2_1'))",
        "MA(10,'iv012_1_2_1')", "MDIFF(60,'iv012_2_3_1')",
        "MADiff(120,MOD('oi039_2_3_1','tc012_5_5_10_1'))"
    ]
    dependencies = [eval(formula)._dependency for formula in factors_list]
    dependencies = list(itertools.chain.from_iterable(dependencies))

    max_windows = [eval(formula)._window for formula in factors_list]
    max_windows = np.array(max_windows).max()

    end_date = advanceDateByCalendar('china.sse', datetime.datetime.now(),
                                     '-{0}b'.format(0)).strftime('%Y-%m-%d')

    begin_date = advanceDateByCalendar(
        'china.sse', end_date, '-{0}b'.format(days)).strftime('%Y-%m-%d')

    start_date = advanceDateByCalendar(
        'china.sse', begin_date,
        '-{0}b'.format(max_windows)).strftime('%Y-%m-%d')

    pdb.set_trace()
    res = load_data_from_dolphin(instruments=instruments,
                                 start_date=start_date,
                                 end_date=end_date)

    impulse = ['i012', 'i009', 'i007']
    ### 方法1 计算指定依赖基础字段
    factors_data1 = Impulse(dependencies).batch(data=res)

    ### 方法2,计算指定的依赖基础字段包
    iactuator = Iactuator(k_split=32, impulse=impulse)
    factors_data2 = iactuator.calculate(total_data=res)

    columns = factors_data1.columns
    for col in columns:
        mask1 = np.isclose(factors_data1[col],
                           factors_data2[col],
                           atol=1e-8,
                           equal_nan=True)
        if mask1.all():
            print("{0} pass".format(col))
        else:
            print("{0} error".format(col))


def impulse_factors2(instruments, method, task_id):
    # 读取已经生成数据
    start_date, end_date = get_dates(method)

    begin_date = advanceDateByCalendar('china.sse', start_date,
                                       '-{0}b'.format(2)).strftime('%Y-%m-%d')

    res2 = load_data_from_dolphin(instruments=instruments,
                                  start_date=begin_date,
                                  end_date=end_date)

    res3 = load_data_from_feather(instruments=instruments,
                                  method=method,
                                  rootid=task_id)

    impulses = [
        'i001', 'i002', 'i004', 'i005', 'i009', 'i010', 'i011', 'i012', 'i014'
    ]

    for impulse in impulses:
        print('--------{0}-------->\n'.format(impulse))
        factors_data1 = pd.read_feather(
            os.path.join(base_path, method, instruments, "factors",
                         "{0}_factors.feather".format(impulse)))
        factors_data1 = factors_data1.set_index(['trade_time', 'code'])
        iactuator = Iactuator(k_split=32, impulse=[impulse])
        factors_data2 = iactuator.calculate(total_data=res2)
        factors_data3 = iactuator.calculate(total_data=res3)
        common_index = factors_data1.index.intersection(
            factors_data2.index).intersection(factors_data3.index)
        common_columns = factors_data1.columns.intersection(
            factors_data2.columns).intersection(factors_data3.columns)
        if common_columns.empty:
            print("两个 DataFrame 没有共同的列名，无法比较。")
        else:
            # 使用 .loc 只选择公共的行和列，这会返回两个形状完全相同且对齐的 DataFrame
            # 这一步非常高效，因为它只是视图（view）或者一个子集的拷贝（copy）
            df1_aligned = factors_data1.loc[common_index, common_columns]
            df2_aligned = factors_data2.loc[common_index, common_columns]
            df3_aligned = factors_data3.loc[common_index, common_columns]

            mask1 = df1_aligned.groupby(level='code').cumcount() >= 240
            df1_trimmed = df1_aligned[mask1]

            mask2 = df2_aligned.groupby(level='code').cumcount() >= 240
            df2_trimmed = df1_aligned[mask2]

            mask3 = df3_aligned.groupby(level='code').cumcount() >= 240
            df3_trimmed = df1_aligned[mask3]

            comparison1_2 = np.isclose(df1_trimmed,
                                       df2_trimmed,
                                       atol=1e-8,
                                       equal_nan=True)
            comparison2_3 = np.isclose(df2_trimmed,
                                       df3_trimmed,
                                       atol=1e-8,
                                       equal_nan=True)

            final_comparison_result = comparison1_2 & comparison2_3
            comparison_df = pd.DataFrame(final_comparison_result,
                                         index=df1_trimmed.index,
                                         columns=df1_trimmed.columns)
            all_pass = comparison_df.all(axis=0)
            for col, passed in all_pass.items():
                if passed:
                    print(f"{col} pass")
                else:
                    print(f"{col} error")


def impulse_factors3(instruments, method, task_id):
    start_date, end_date = get_dates(method)

    begin_date = advanceDateByCalendar('china.sse', start_date,
                                       '-{0}b'.format(2)).strftime('%Y-%m-%d')
    pdb.set_trace()
    # res = load_data_from_feather(instruments=instruments,
    #                              method=method,
    #                              rootid=task_id)
    res = load_data_from_dolphin(instruments=instruments,
                                 start_date=begin_date,
                                 end_date=end_date)

    impulses = {'i012': ['oi039_1_2_1']}
    for impulse, dependencies in impulses.items():
        factors_data1 = pd.read_feather(
            os.path.join(base_path, method, instruments, "factors",
                         "{0}_factors.feather".format(impulse)))
        factors_data1 = factors_data1.set_index(['trade_time', 'code'])
        pdb.set_trace()
        factors_data2 = Impulse(dependencies).batch(data=res)
        common_index = factors_data1.index.intersection(factors_data2.index)

        df1_aligned = factors_data1.loc[common_index, dependencies]
        df2_aligned = factors_data2.loc[common_index, dependencies]

        mask1 = df1_aligned.groupby(level='code').cumcount() >= 240
        df1_trimmed = df1_aligned[mask1]

        mask2 = df2_aligned.groupby(level='code').cumcount() >= 240
        df2_trimmed = df1_aligned[mask2]

        comparison_result = np.isclose(df1_trimmed,
                                       df2_trimmed,
                                       atol=1e-8,
                                       equal_nan=True)
        comparison_df = pd.DataFrame(comparison_result,
                                     index=df1_trimmed.index,
                                     columns=df1_trimmed.columns)
        all_pass = comparison_df.all(axis=0)
        for col, passed in all_pass.items():
            if passed:
                print(f"{col} pass")
            else:
                print(f"{col} error")


def _price_diff_metrics(research_market, trader_market, tick_size,
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


def _relative_diff_metrics(research_market, trader_market, rel_fields):
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


##基础数据对比
def impulse_market_data(instruments, tick_size=1):  # 最小价格变动

    adjusted_method = None#'pcr'
    price_fields = ['open', 'high', 'low', 'close', 'vwap']
    rel_fiedls = ["volume", "value", "openint"]
    common_columns = price_fields + rel_fiedls
    begin_time = datetime.datetime(2026, 4, 3)
    end_time = datetime.datetime(2026, 5, 13)
    research_market = fetch_main_market(begin_date=begin_time,
                                        end_date=end_time,
                                        codes=[INSTRUMENTS_CODES[instruments]],
                                        method=adjusted_method,
                                        keep_symbol=True)
    research_market['trade_time'] = pd.to_datetime(
        research_market['trade_time'])
    research_market = research_market.set_index(['trade_time', 'code'])

    trader_market = fetch_trader_market1(begin_time=begin_time,
                                         end_time=end_time,
                                         code=INSTRUMENTS_CODES[instruments],
                                         adjusted_method=adjusted_method)
    trader_market['trade_time'] = pd.to_datetime(trader_market['trade_time'])
    trader_market = trader_market.set_index(['trade_time', 'code'])

    common_index = research_market.index.intersection(trader_market.index)

    research_market = research_market.loc[common_index, common_columns]
    trader_market = trader_market.loc[common_index, common_columns]
    pdb.set_trace()

    ## 之前数据弄错了， 暂时设置为一样。目前已经修改和文华财经一致
    trader_market['value'] = research_market['value']
    trader_market['volume'] = research_market['volume']
    trader_market['vwap'] = research_market['vwap']

    pdb.set_trace()
    ## 价格差异指标
    price_metrics = _price_diff_metrics(research_market=research_market,
                                        trader_market=trader_market,
                                        tick_size=tick_size,
                                        price_fields=price_fields)

    rel_metrics = _relative_diff_metrics(research_market=research_market,
                                         trader_market=trader_market,
                                         rel_fields=rel_fiedls)
    price_metrics = pd.DataFrame(price_metrics)
    rel_metrics = pd.DataFrame(rel_metrics)
    pdb.set_trace()
    print('-->')


impulse_factors2(instruments='rbb', method='bicso2', task_id='113001')
