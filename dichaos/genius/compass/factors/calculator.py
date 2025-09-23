import pdb
import numpy as np
from alphacopilot.calendars.api import advanceDateByCalendar
import lumina.impulse.v001 as v001
from .extractor import *


def slice_with_metadata(series, start=None, end=None):
    sliced_series = series.loc[start:end]
    if hasattr(series, 'desc'):
        sliced_series.desc = series.desc
    return sliced_series


def create_chg(codes, begin_date, end_date, offset=1):
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(2))
    finish_date = advanceDateByCalendar('china.sse', end_date,
                                        '{0}b'.format(2))

    market_data = fetch_main_daily(begin_date=start_date,
                                   end_date=finish_date,
                                   codes=codes,
                                   columns=[
                                       'trade_date', 'code', 'openPrice',
                                       'highestPrice', 'lowestPrice',
                                       'closePrice', 'turnoverVol',
                                       'turnoverValue', 'openInt'
                                   ])
    market_data = market_data.set_index(['trade_date', 'code']).unstack()
    returns = np.log(market_data['close'] /
                     market_data['close'].shift(1))
    returns = returns.shift(offset)
    returns.name = 'returns'
    returns = returns.loc[begin_date:end_date]
    returns.index = pd.to_datetime(returns.index).strftime('%Y-%m-%d')
    return returns.loc[begin_date:end_date].to_dict()


def create_factors(name, keys, kl_pd, begin_date, end_date, factor_res):
    impulse = getattr(v001, 'Impulse{0}'.format(name))(keys=keys)
    factors = impulse.calc_impulse(kl_pd=kl_pd)
    ## 对齐因子
    for k, v in factors.items():
        factors[k] = slice_with_metadata(series=v,
                                         start=begin_date,
                                         end=end_date)
    factor_res.update(factors)


def create_posflow(begin_date, end_date, codes, window, category, **kwargs):
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))
    member_positions = fetch_member_positions(begin_date=start_date,
                                              end_date=end_date,
                                              codes=codes,
                                              category=category)
    openint_position = fetch_main_daily(
        begin_date=start_date,
        end_date=end_date,
        codes=codes,
        columns=['trade_date', 'code', 'openInt'])

    position_data = member_positions.merge(openint_position,
                                           on=['trade_date', 'code'])
    position_data = position_data.set_index(['trade_date', 'code']).unstack()
    create_factors(name="Mo001",
                   keys=[(1, 1, 1)],
                   kl_pd=position_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mo002',
                   keys=[(1, 1, 1)],
                   kl_pd=position_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mo003',
                   keys=[(1, 1, 1)],
                   kl_pd=position_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mo004',
                   keys=[(1, 1, 1)],
                   kl_pd=position_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mo005',
                   keys=[(1, 1, 1)],
                   kl_pd=position_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mo007',
                   keys=[(1, 1, 1)],
                   kl_pd=position_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mo008',
                   keys=[(1, 1, 1)],
                   kl_pd=position_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mo009',
                   keys=[(1, 1, 1)],
                   kl_pd=position_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    return factor_res


def create_kline(begin_date, end_date, codes, window, **kwargs):
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))

    market_data = fetch_main_daily(begin_date=start_date,
                                   end_date=end_date,
                                   codes=codes,
                                   columns=[
                                       'trade_date', 'code', 'openPrice',
                                       'highestPrice', 'lowestPrice',
                                       'closePrice', 'turnoverVol',
                                       'turnoverValue', 'openInt'
                                   ])
    market_data = market_data.set_index(['trade_date', 'code'])
    columns = market_data.columns
    for col in columns:
        f = market_data[col]
        f.name = col
        f.desc = col
        factor_res[col] = f
    return factor_res


def create_indictor(begin_date, end_date, codes, window, **kwargs):
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))
    market_data = fetch_main_daily(begin_date=start_date,
                                   end_date=end_date,
                                   codes=codes,
                                   columns=[
                                       'trade_date', 'code', 'openPrice',
                                       'highestPrice', 'lowestPrice',
                                       'closePrice', 'turnoverVol',
                                       'turnoverValue', 'openInt'
                                   ])
    market_data = market_data.drop_duplicates(
        subset=['trade_date', 'code']).set_index(['trade_date',
                                                  'code']).unstack()

    create_factors(name="In001",
                   keys=[(1, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="In002",
                   keys=[(1, 5, 1), (1, 10, 1), (1, 20, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="In003",
                   keys=[(1, 12, 1), (1, 26, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="In004",
                   keys=[(1, 14, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name="In005",
                   keys=[(1, 12, 26, 9, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name="In006",
                   keys=[(1, 20, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name="In007",
                   keys=[(1, 7, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name="In008",
                   keys=[(1, 14, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name="In009",
                   keys=[(1, 1, 1), (1, 5, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name="In010",
                   keys=[(1, 14, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name="In011",
                   keys=[(1, 1, 1)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    return factor_res


def create_moneyflow(begin_date, end_date, codes, category, window, **kwargs):
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))
    moneyflow_data = fetch_moneyflow_data(begin_date=start_date,
                                          end_date=end_date,
                                          codes=codes,
                                          category=category)
    returns_data = fetch_main_returns(begin_date=start_date,
                                      end_date=end_date,
                                      codes=[codes])
    returns_data = returns_data.rename({'returns': 'ret'}, axis=1)
    #moneyflow_data = moneyflow_data.merge(returns_data,
    #                                      on=['trade_date', 'code']).unstack()
    moneyflow_data = pd.concat([moneyflow_data, returns_data],
                               axis=1).unstack()
    create_factors(name='Mf001',
                   keys=[(10, 30, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf002',
                   keys=[(10, 30, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf003',
                   keys=[(20, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf004',
                   keys=[(50, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf005',
                   keys=[(5, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf006',
                   keys=[(5, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf007',
                   keys=[(0, )],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf008',
                   keys=[(20, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf009',
                   keys=[(20, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf010',
                   keys=[(20, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf011',
                   keys=[(20, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf012',
                   keys=[(10, 35, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf013',
                   keys=[(10, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf014',
                   keys=[(20, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf015',
                   keys=[(0, )],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf016',
                   keys=[(10, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    # pdb.set_trace()
    create_factors(name='Mf017',
                   keys=[(5, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Mf018',
                   keys=[(20, 1)],
                   kl_pd=moneyflow_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    return factor_res


def create_heat(begin_date, end_date, codes, category, window, **kwargs):
    assert category in ['xueqiu', 'guba']
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))
    heat_data = fetch_heat_data(begin_date=start_date,
                                end_date=end_date,
                                codes=codes)
    create_factors(name='Ht001',
                   keys=[(category, 1, 5, 1), (category, 1, 10, 1),
                         (category, 1, 20, 1)],
                   kl_pd=heat_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht002',
                   keys=[(category, 1, 5, 1), (category, 1, 10, 1),
                         (category, 1, 20, 1)],
                   kl_pd=heat_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht003',
                   keys=[(category, 1, 14, 1)],
                   kl_pd=heat_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht004',
                   keys=[(category, 1, 12, 26, 9, 1)],
                   kl_pd=heat_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht005',
                   keys=[(category, 1, 20, 1)],
                   kl_pd=heat_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht006',
                   keys=[(category, 1, 7, 1)],
                   kl_pd=heat_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    return factor_res


### 计算筹码
def create_chip(begin_date, end_date, codes, window, **kwargs):
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))
    market_data = fetch_main_daily(begin_date=start_date,
                                   end_date=end_date,
                                   codes=codes,
                                   columns=[
                                       'trade_date', 'code', 'openPrice',
                                       'highestPrice', 'lowestPrice',
                                       'closePrice', 'turnoverVol',
                                       'turnoverValue', 'openInt'
                                   ])
    market_data = market_data.drop_duplicates(
        subset=['trade_date', 'code']).set_index(['trade_date',
                                                  'code'])  #.unstack()

    chip_data = calculator_chip(market_data=market_data)
    #market_data['chip_data'] = chip_data
    res = {}
    for col in market_data.columns:
        res[col] = market_data[col].unstack()
    res['chip_data'] = chip_data
    market_data = res
    create_factors(name="Cp001",
                   keys=[()],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp002",
                   keys=[()],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp003",
                   keys=[(0.1, ), (0.2, )],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp004",
                   keys=[(0.8, ), (0.9, )],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp005",
                   keys=[(0.8, 5), (0.9, 5)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp006",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp007",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp008",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp009",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp010",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp011",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp012",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp013",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp014",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp015",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    #create_factors(name="Cp016",
    #               keys=[(1, 0), (5, 0)],
    #               kl_pd=market_data,
    #               begin_date=begin_date,
    #               end_date=end_date,
    #               factor_res=factor_res)

    create_factors(name="Cp017",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name="Cp018",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp019",
                   keys=[(1, 0), (5, 0)],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name="Cp020",
                   keys=[()],
                   kl_pd=market_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    return factor_res


###辅助函数
def fast_chip_simulation(lows, highs, turnovers, price_bins, decay_factor):
    """
    使用 Numba 加速的筹码分布计算核心循环。
    **最终修正版：将每日换手筹码均匀分布在当日价格区间(low-high)内。**
    """
    num_days = len(lows)
    num_bins = len(price_bins)
    chip_distribution = np.zeros(num_bins, dtype=np.float64)

    # 首次交易，将筹码均匀分布在当天的价格区间
    start_index = np.searchsorted(price_bins, lows[0], side='right') - 1
    end_index = np.searchsorted(price_bins, highs[0], side='right') - 1

    if 0 <= start_index < num_bins and 0 <= end_index < num_bins and start_index <= end_index:
        num_bins_in_range = end_index - start_index + 1
        chip_distribution[start_index:end_index + 1] = 1.0 / num_bins_in_range

    # 从第二天开始迭代
    for i in range(1, num_days):
        turnover = turnovers[i]
        low_price = lows[i]
        high_price = highs[i]

        # 1. 对昨天的【全部】筹码应用时间衰减
        chip_distribution *= decay_factor

        # 2. 计算换手
        exchanged_chips_total = chip_distribution.sum() * turnover
        chip_distribution *= (1 - turnover)

        # --- 核心逻辑重构 ---
        # 3. 将换手的筹码【均匀分布】到当天新的成本【区间】
        start_index = np.searchsorted(price_bins, low_price, side='right') - 1
        end_index = np.searchsorted(price_bins, high_price, side='right') - 1

        if 0 <= start_index < num_bins and 0 <= end_index < num_bins and start_index <= end_index:
            num_bins_in_range = end_index - start_index + 1
            chips_per_bin = exchanged_chips_total / num_bins_in_range
            chip_distribution[start_index:end_index + 1] += chips_per_bin
        # --- 重构结束 ---

    return chip_distribution


def calculator_chip(market_data):
    decay_factor = 0.999
    turnover_damping_factor = 0.15

    hist_data = market_data.copy()

    hist_data['turnover'] = ((hist_data['volume'] /
                              (hist_data['openint'] + 1e-9)) *
                             turnover_damping_factor).clip(0, 1)

    # --- 参数传递修改 ---
    # 准备 Numpy 数组，现在需要 low 和 high
    lows_np = hist_data['low'].to_numpy()
    highs_np = hist_data['high'].to_numpy()
    turnovers_np = hist_data['turnover'].to_numpy()
    # --- 修改结束 ---

    # 创建价格网格
    price_min = hist_data['low'].min()
    price_max = hist_data['high'].max()
    price_bins_np = np.linspace(price_min, price_max, 201)

    # --- 函数调用修改 ---
    final_chips_np = fast_chip_simulation(lows_np, highs_np, turnovers_np,
                                          price_bins_np, decay_factor)
    # --- 修改结束 ---

    total_chips = final_chips_np.sum()
    if total_chips > 0:
        final_chips_np = (final_chips_np / total_chips) * 100

    result_df = pd.DataFrame({'price': price_bins_np, 'chips': final_chips_np})
    result_df = result_df[result_df['chips'] > 0.1]  # 适当提高阈值以获得更清晰的图

    return result_df
