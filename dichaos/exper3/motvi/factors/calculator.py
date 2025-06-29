import pdb
from alphacopilot.calendars.api import advanceDateByCalendar
import lumina.impulse.v001 as v001
from motvi.factors.extractor import fetch_member_positions, fetch_main_daily


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
    market_data = market_data.set_index(['trade_date', 'code']).unstack()

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
