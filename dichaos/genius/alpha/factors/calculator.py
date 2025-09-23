import numpy as np
from numba import jit
from alphacopilot.calendars.api import advanceDateByCalendar
from .extractor import *
import lumina.impulse.v001 as v001


def slice_with_metadata(series, start=None, end=None):
    sliced_series = series.loc[start:end]
    if hasattr(series, 'desc'):
        sliced_series.desc = series.desc
    return sliced_series


### T 期特征--> T+1期vwap价格与T+1期vwap价格 算收益率
def create_return(codes, begin_date, end_date):
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(2))
    finish_date = advanceDateByCalendar('china.sse', end_date,
                                        '{0}b'.format(2))
    market_data = fetch_main_daily(begin_date=start_date.strftime('%Y-%m-%d'),
                                   end_date=finish_date.strftime('%Y-%m-%d'),
                                   codes=codes,
                                   columns=[
                                       'openPrice', 'highestPrice',
                                       'lowestPrice', 'closePrice',
                                       'turnoverVol', 'turnoverValue'
                                   ])

    market_data['vwap'] = market_data['amount'] / market_data[
        'volume']  ## 出现了成交量 成交额为0
    market_data['returns'] = np.log(market_data['vwap'].shift(2) /
                                    market_data['vwap'].shift(1)).shift(-2)
    returns = market_data['returns']
    returns.name = 'returns'
    returns = returns.loc[begin_date:end_date]
    returns.index = pd.to_datetime(returns.index).strftime('%Y-%m-%d')
    return returns.loc[begin_date:end_date].to_dict()


def create_chg(codes, begin_date, end_date):
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(2))
    finish_date = advanceDateByCalendar('china.sse', end_date,
                                        '{0}b'.format(2))
    market_data = fetch_main_daily(begin_date=start_date.strftime('%Y-%m-%d'),
                                   end_date=finish_date.strftime('%Y-%m-%d'),
                                   codes=codes,
                                   columns=[
                                       'openPrice', 'highestPrice',
                                       'lowestPrice', 'closePrice', 'ret'
                                   ])
    returns = market_data['ret']
    returns.name = 'returns'
    returns = returns.loc[begin_date:end_date]
    returns.index = pd.to_datetime(returns.index).strftime('%Y-%m-%d')
    return returns.loc[begin_date:end_date].to_dict()


def create_factors(name, keys, kl_pd, begin_date, end_date, factor_res):
    impulse = getattr(v001, 'Impulse{0}'.format(name))(keys=keys)
    factors = impulse.calc_impulse(kl_pd=kl_pd)
    ## 对齐因子
    new_factors = {}
    for k, v in factors.items():
        series = slice_with_metadata(series=v, start=begin_date, end=end_date)
        if not series.empty:
            new_factors[k] = series
    factor_res.update(new_factors)


def create_kline(begin_date, end_date, codes, window, **kwargs):
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))

    market_data = fetch_main_daily(begin_date=start_date.strftime('%Y-%m-%d'),
                                   end_date=end_date,
                                   codes=codes,
                                   columns=[
                                       'openPrice', 'highestPrice',
                                       'lowestPrice', 'closePrice',
                                       'turnoverVol', 'turnoverValue'
                                   ])
    for col in market_data.keys():
        dt = market_data[col]
        dt.name = col
        dt.desc = col
        market_data[col] = dt
    return factor_res


def create_indictor(begin_date, end_date, codes, window, **kwargs):
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))
    market_data = fetch_main_daily(begin_date=start_date.strftime('%Y-%m-%d'),
                                   end_date=end_date,
                                   codes=codes,
                                   columns=[
                                       'openPrice', 'highestPrice',
                                       'lowestPrice', 'closePrice',
                                       'turnoverVol', 'turnoverValue'
                                   ])
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


def create_moneyflow(begin_date, end_date, codes, window, **kwargs):
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))
    moneyflow_data = fetch_moneyflow_data(begin_date=start_date,
                                          end_date=end_date,
                                          codes=codes)
    specify_data = fetch_specify_data(
        begin_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date,
        codes=codes,
        columns=['ret'])
    returns_data = specify_data['ret'].stack()
    returns_data.name = 'ret'
    moneyflow_data = moneyflow_data.merge(returns_data,
                                          on=['trade_date', 'code'])
    moneyflow_data = moneyflow_data.set_index(['trade_date', 'code']).unstack()
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
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window * 2))
    clouto_data = fetch_clouto_data(begin_date=start_date,
                                    end_date=end_date,
                                    codes=codes)
    create_factors(name='Ht001',
                   keys=[(category, 1, 5, 1), (category, 1, 5, 0),
                         (category, 1, 10, 1)],
                   kl_pd=clouto_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht002',
                   keys=[(category, 1, 5, 1), (category, 1, 5, 0),
                         (category, 1, 10, 1)],
                   kl_pd=clouto_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name='Ht003',
                   keys=[(category, 1, 4, 1)],
                   kl_pd=clouto_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht004',
                   keys=[(category, 1, 6, 13, 5, 1)],
                   kl_pd=clouto_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht005',
                   keys=[(category, 1, 10, 1)],
                   kl_pd=clouto_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht006',
                   keys=[(category, 1, 7, 1)],
                   kl_pd=clouto_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    return factor_res


def create_hotmoney(begin_date, end_date, codes, window, **kwargs):
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))
    hotmoney_data = fetch_hotmoney(begin_date=start_date.strftime('%Y-%m-%d'),
                                   end_date=end_date,
                                   codes=codes)
    ### 针对不同因子对空值的处理
    hotmoney_data1 = hotmoney_data.copy()
    hotmoney_data2 = hotmoney_data.fillna(0)
    hotmoney_data3 = hotmoney_data.ffill()

    create_factors(name='Hm001',
                   keys=[(1, 1, 1), (1, 2, 1)],
                   kl_pd=hotmoney_data2,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name='Hm002',
                   keys=[(1, 1, 1), (1, 5, 1)],
                   kl_pd=hotmoney_data2,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name='Hm003',
                   keys=[(1, 1, 1), (1, 3, 1), (1, 5, 1)],
                   kl_pd=hotmoney_data2,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name='Hm004',
                   keys=[(1, 1, 1), (1, 3, 1), (1, 5, 1)],
                   kl_pd=hotmoney_data2,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name='Hm005',
                   keys=[(1, 1, 1), (1, 3, 1), (1, 5, 1)],
                   kl_pd=hotmoney_data2,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name='Hm006',
                   keys=[(1, 1, 1), (1, 3, 1), (1, 5, 1)],
                   kl_pd=hotmoney_data2,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name='Hm007',
                   keys=[(1, 1, 1), (1, 3, 1), (1, 5, 1)],
                   kl_pd=hotmoney_data2,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name='Hm008',
                   keys=[(1, 1, 1), (1, 3, 1), (1, 5, 1)],
                   kl_pd=hotmoney_data2,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name='Hm009',
                   keys=[(1, 1, 1), (1, 3, 1), (1, 5, 1)],
                   kl_pd=hotmoney_data2,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)

    create_factors(name='Hm010',
                   keys=[(1, 1, 1), (1, 3, 1), (1, 5, 1)],
                   kl_pd=hotmoney_data2,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    return factor_res


def create_chip(begin_date, end_date, codes, window, **kwargs):
    factor_res = {}
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(window))
    market_data = fetch_main_daily(begin_date=start_date.strftime("%Y-%m-%d"),
                                   end_date=end_date,
                                   codes=codes,
                                   columns=[
                                       'openPrice', 'highestPrice',
                                       'lowestPrice', 'closePrice',
                                       'turnoverVol', 'turnoverValue',
                                       'turnoverRate'
                                   ])
    chip_data = calculator_chip(market_data=market_data)
    market_data['chip_data'] = chip_data

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
#@jit(nopython=True)
def fast_chip_simulation(prices, turnovers, price_bins, decay_factor):
    """
    使用 Numba 加速的筹-码分布计算核心循环。
    这里的所有操作都基于 Numpy 数组，以获得最高性能。
    """
    num_days = len(prices)
    num_bins = len(price_bins)
    chip_distribution = np.zeros(num_bins, dtype=np.float64)
    # 首次交易，假设全部筹码在当天成交
    initial_price_index = np.searchsorted(price_bins, prices[0],
                                          side='right') - 1
    if 0 <= initial_price_index < num_bins:
        chip_distribution[initial_price_index] = 1.0
    # 从第二天开始迭代
    for i in range(1, num_days):
        turnover = turnovers[i]
        avg_price = prices[i]

        # 1. 计算衰减和换手
        # a. 未换手的筹码进行衰减
        remaining_chips = chip_distribution * (1 - turnover)
        # b. 换手的筹码进行更强的衰减（或视为完全移除并重新分配）
        exchanged_chips_total = chip_distribution.sum() * turnover

        # 应用衰减
        chip_distribution = remaining_chips * decay_factor

        # 2. 将换手的筹码添加到新成本区
        price_index = np.searchsorted(price_bins, avg_price, side='right') - 1
        if 0 <= price_index < num_bins:
            chip_distribution[price_index] += exchanged_chips_total

    return chip_distribution


def calculator_chip(market_data):
    res = []
    decay_factor = 0.999
    for k, v in market_data.items():
        v1 = v.stack()
        v1.name = k
        res.append(v1)
    hist_data = pd.concat(res, axis=1)
    hist_data['avg_price'] = (hist_data['low'] + hist_data['high']) / 2
    # 1. 准备 Numpy 数组
    avg_prices_np = hist_data['avg_price'].to_numpy()
    turnovers_np = hist_data['turnover'].to_numpy()

    # 2. 创建价格网格
    price_min = hist_data['low'].min()
    price_max = hist_data['high'].max()
    price_bins_np = np.linspace(price_min, price_max, 201)

    final_chips_np = fast_chip_simulation(avg_prices_np, turnovers_np,
                                          price_bins_np, decay_factor)

    total_chips = final_chips_np.sum()
    if total_chips > 0:
        final_chips_np = (final_chips_np / total_chips) * 100

    result_df = pd.DataFrame({'price': price_bins_np, 'chips': final_chips_np})
    result_df = result_df[result_df['chips'] > 0.001]
    return result_df
