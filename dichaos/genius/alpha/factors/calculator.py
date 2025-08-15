from alphacopilot.calendars.api import advanceDateByCalendar
from .extractor import *
import lumina.impulse.v001 as v001


def slice_with_metadata(series, start=None, end=None):
    sliced_series = series.loc[start:end]
    if hasattr(series, 'desc'):
        sliced_series.desc = series.desc
    return sliced_series


def create_factors(name, keys, kl_pd, begin_date, end_date, factor_res):
    impulse = getattr(v001, 'Impulse{0}'.format(name))(keys=keys)
    factors = impulse.calc_impulse(kl_pd=kl_pd)
    ## 对齐因子
    for k, v in factors.items():
        factors[k] = slice_with_metadata(series=v,
                                         start=begin_date,
                                         end=end_date)
    factor_res.update(factors)


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
                                       '-{0}b'.format(window))
    clouto_data = fetch_clouto_data(begin_date=start_date,
                                    end_date=end_date,
                                    codes=codes)
    create_factors(name='Ht001',
                   keys=[(category, 1, 5, 1), (category, 1, 10, 1),
                         (category, 1, 20, 1)],
                   kl_pd=clouto_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht002',
                   keys=[(category, 1, 5, 1), (category, 1, 10, 1),
                         (category, 1, 20, 1)],
                   kl_pd=clouto_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht003',
                   keys=[(category, 1, 14, 1)],
                   kl_pd=clouto_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht004',
                   keys=[(category, 1, 12, 26, 9, 1)],
                   kl_pd=clouto_data,
                   begin_date=begin_date,
                   end_date=end_date,
                   factor_res=factor_res)
    create_factors(name='Ht005',
                   keys=[(category, 1, 20, 1)],
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
