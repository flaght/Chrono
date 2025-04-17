import pdb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from alphacopilot.api.data import RetrievalAPI, ddb_tools, DDBAPI
from kdutils.ttimes import get_dates


def fetch_daily(begin_date, end_date, universe):
    data = RetrievalAPI.get_data_by_map(columns=[
        'dummy120_fst',
        universe,
        'lowestPrice',
        'openPrice',
        'closePrice',
        'highestPrice',
        'turnoverVol',
        'turnoverValue',
        'adjFactorPrice',
        'adjFactorVol',
    ],
                                        begin_date=begin_date,
                                        end_date=end_date,
                                        method='ddb')

    lowestPrice = (data['lowestPrice'] * data['dummy120_fst'] *
                   data[universe] * data['adjFactorPrice']).stack()
    lowestPrice.name = 'low'

    openPrice = (data['openPrice'] * data['dummy120_fst'] * data[universe] *
                 data['adjFactorPrice']).stack()
    openPrice.name = 'open'

    closePrice = (data['closePrice'] * data['dummy120_fst'] * data[universe] *
                  data['adjFactorPrice']).stack()
    closePrice.name = 'close'

    highestPrice = (data['highestPrice'] * data['dummy120_fst'] *
                    data[universe] * data['adjFactorPrice']).stack()
    highestPrice.name = 'high'

    turnoverVol = (data['turnoverVol'] * data['dummy120_fst'] *
                   data[universe] * data['adjFactorVol']).stack()
    turnoverVol.name = 'volume'

    turnoverValue = (data['turnoverValue'] * data['dummy120_fst'] *
                     data[universe]).stack()
    turnoverValue.name = 'value'

    data = pd.concat([
        lowestPrice, openPrice, closePrice, highestPrice, turnoverVol,
        turnoverValue
    ],
                     axis=1)
    data = data.unstack()
    low = data['low'].fillna(method='ffill').fillna(0).unstack()
    low.name = 'low'
    close = data['close'].fillna(method='ffill').fillna(0).unstack()
    close.name = 'close'
    high = data['high'].fillna(method='ffill').fillna(0).unstack()
    high.name = 'high'
    open = data['open'].fillna(method='ffill').fillna(0).unstack()
    open.name = 'open'

    volume = data['volume'].fillna(0).unstack()
    volume.name = 'volume'

    value = data['value'].fillna(0).unstack()
    value.name = 'value'

    price = data['value'].fillna(0) / data['volume'].fillna(0)
    price = price.fillna(method='ffill').fillna(0).unstack()
    price.name = 'price'

    data = pd.concat([low, close, high, open, volume, value, price], axis=1)
    return data


def fetch_market(begin_date, end_date, universe):
    data = RetrievalAPI.get_data_by_map(columns=[
        'dummy120_fst',
        universe,
        'lowestPrice',
        'openPrice',
        'closePrice',
        'highestPrice',
        'turnoverVol',
        'turnoverValue',
        'adjFactorPrice',
        'adjFactorVol',
    ],
                                        begin_date=begin_date,
                                        end_date=end_date,
                                        method='ddb')

    lowestPrice = (data['lowestPrice'] * data['dummy120_fst'] *
                   data[universe] * data['adjFactorPrice']).stack()
    lowestPrice.name = 'low'

    openPrice = (data['openPrice'] * data['dummy120_fst'] * data[universe] *
                 data['adjFactorPrice']).stack()
    openPrice.name = 'open'

    closePrice = (data['closePrice'] * data['dummy120_fst'] * data[universe] *
                  data['adjFactorPrice']).stack()
    closePrice.name = 'close'

    highestPrice = (data['highestPrice'] * data['dummy120_fst'] *
                    data[universe] * data['adjFactorPrice']).stack()
    highestPrice.name = 'high'

    turnoverVol = (data['turnoverVol'] * data['dummy120_fst'] *
                   data[universe] * data['adjFactorVol']).stack()
    turnoverVol.name = 'volume'

    turnoverValue = (data['turnoverValue'] * data['dummy120_fst'] *
                     data[universe]).stack()
    turnoverValue.name = 'value'

    data = pd.concat([
        lowestPrice, openPrice, closePrice, highestPrice, turnoverVol,
        turnoverValue
    ],
                     axis=1)
    data = data.unstack()
    low = data['low'].fillna(method='ffill').fillna(0).unstack()
    low.name = 'low'
    close = data['close'].fillna(method='ffill').fillna(0).unstack()
    close.name = 'close'
    high = data['high'].fillna(method='ffill').fillna(0).unstack()
    high.name = 'high'
    open = data['open'].fillna(method='ffill').fillna(0).unstack()
    open.name = 'open'

    volume = data['volume'].fillna(0).unstack()
    volume.name = 'volume'

    value = data['value'].fillna(0).unstack()
    value.name = 'value'

    price = data['value'].fillna(0) / data['volume'].fillna(0)
    price = price.fillna(method='ffill').fillna(0).unstack()
    price.name = 'price'

    ####价格使用前置价格 + 0 填充， 成交量使用0
    data = pd.concat([low, close, high, open, volume, value, price], axis=1)
    res = []
    grouped = data.groupby(level=['code'])
    for k, v in grouped:
        print(k)
        v = v.sort_index()
        #v['oopen'] = v['open']
        v[['low', 'close', 'high',
           'open']] = v[['low', 'close', 'high', 'open'
                         ]] / v[['low', 'close', 'high', 'open']].max()
        v[['volume']] = v[['volume']] / v[['volume']].max()
        v[['value']] = v[['value']] / v[['value']].max()
        v['dopen'] = v['open'] - v['open'].shift(1)
        v['dclose'] = v['close'] - v['close'].shift(1)
        v['dhigh'] = v['high'] - v['high'].shift(1)
        v['dlow'] = v['low'] - v['high'].shift(1)
        v['dvolume'] = v['volume'] - v['volume'].shift(1)
        v['dvalue'] = v['value'] - v['value'].shift(1)
        v = v.reset_index()
        v = v.rename(columns={'trade_date': 'date'})
        v = v.dropna()
        res.append(v.set_index(['date', 'code']))
    market_data = pd.concat(res, axis=0)
    return market_data.sort_index()


def fetch_risk_exposure(begin_date, end_date):
    clause_list1 = ddb_tools.to_format('flag', '==', 1)
    clause_list2 = ddb_tools.to_format('trade_date', '>=',
                                       ddb_tools.convert_date(begin_date))
    clause_list3 = ddb_tools.to_format('trade_date', '<=',
                                       ddb_tools.convert_date(end_date))
    cusomize_api = DDBAPI.cusomize_api()
    exposure_data = cusomize_api.custom(
        table='risk_exposure',
        clause_list=[clause_list1, clause_list2, clause_list3],
        format_data=0)
    return exposure_data


def fetch_risk_special(begin_date, end_date):
    clause_list1 = ddb_tools.to_format('flag', '==', 1)
    clause_list2 = ddb_tools.to_format('trade_date', '>=',
                                       ddb_tools.convert_date(begin_date))
    clause_list3 = ddb_tools.to_format('trade_date', '<=',
                                       ddb_tools.convert_date(end_date))
    cusomize_api = DDBAPI.cusomize_api()
    special_data = cusomize_api.custom(
        table='risk_special',
        clause_list=[clause_list1, clause_list2, clause_list3],
        format_data=0)
    return special_data


def fetch_risk_cov_day(begin_date, end_date):
    clause_list1 = ddb_tools.to_format('flag', '==', 1)
    clause_list2 = ddb_tools.to_format('trade_date', '>=',
                                       ddb_tools.convert_date(begin_date))
    clause_list3 = ddb_tools.to_format('trade_date', '<=',
                                       ddb_tools.convert_date(end_date))
    cusomize_api = DDBAPI.cusomize_api()
    special_data = cusomize_api.custom(
        table='risk_cov_day',
        clause_list=[clause_list1, clause_list2, clause_list3],
        format_data=0)
    return special_data


def fetch_risk_cov_short(begin_date, end_date):
    clause_list1 = ddb_tools.to_format('flag', '==', 1)
    clause_list2 = ddb_tools.to_format('trade_date', '>=',
                                       ddb_tools.convert_date(begin_date))
    clause_list3 = ddb_tools.to_format('trade_date', '<=',
                                       ddb_tools.convert_date(end_date))
    cusomize_api = DDBAPI.cusomize_api()
    special_data = cusomize_api.custom(
        table='risk_cov_short',
        clause_list=[clause_list1, clause_list2, clause_list3],
        format_data=0)
    return special_data


def fetch_risk_cov_long(begin_date, end_date):
    clause_list1 = ddb_tools.to_format('flag', '==', 1)
    clause_list2 = ddb_tools.to_format('trade_date', '>=',
                                       ddb_tools.convert_date(begin_date))
    clause_list3 = ddb_tools.to_format('trade_date', '<=',
                                       ddb_tools.convert_date(end_date))
    cusomize_api = DDBAPI.cusomize_api()
    special_data = cusomize_api.custom(
        table='risk_cov_long',
        clause_list=[clause_list1, clause_list2, clause_list3],
        format_data=0)
    return special_data


def fetch_hdetail(universe, horzion):
    clause_list1 = ddb_tools.to_format('status', '==', 1)
    clause_list2 = ddb_tools.to_format('universe', 'in', [universe])
    clause_list3 = ddb_tools.to_format('horizon', 'in', [horzion])
    cusomize_api = DDBAPI.cusomize_api()
    pdb.set_trace()
    factors_data = cusomize_api.custom(
        table='stk_xyfactorinfo',
        columns=['id'],
        clause_list=[clause_list1, clause_list2, clause_list3],
        format_data=1)
    factors_data = factors_data.drop_duplicates(subset=['id'])
    return factors_data['id'].tolist()


def fetch_hfactors(begin_date, end_date, universe, horzion, is_scale=True):
    ids = fetch_hdetail(universe=universe, horzion=horzion)
    pdb.set_trace()
    factors_data = RetrievalAPI.get_factors(begin_date=begin_date,
                                            end_date=end_date,
                                            ids=ids,
                                            freq='D',
                                            format_data=1,
                                            is_debug=True)
    data = RetrievalAPI.get_data_by_map(columns=[
        'dummy120_fst',
        universe,
    ],
                                        begin_date=begin_date,
                                        end_date=end_date,
                                        method='ddb')

    res = []
    for f in factors_data.keys():
        print(f)
        ff = factors_data[f]
        ff = (ff * data['dummy120_fst'] * data[universe]).stack()
        ff.name = f
        res.append(ff)
    factors = pd.concat(res, axis=1)
    factors = factors.unstack().fillna(method='ffill').fillna(0)
    factors = factors.stack()
    ## 标准化
    if is_scale:
        features = factors.columns
        scaler = StandardScaler()
        factors[features] = scaler.fit_transform(factors[features].values)
    return factors


def fetch_chgpct(begin_date, end_date):
    data = RetrievalAPI.get_data_by_map(columns=['ret_o2o', 'dummy120_fst'],
                                        begin_date=begin_date,
                                        end_date=end_date,
                                        method='ddb')
    chg_pct = data['ret_o2o'].unstack()
    chg_pct.name = 'chg_pct'

    vardummy = data['dummy120_fst'].unstack()
    vardummy.name = 'dummy'

    return pd.concat([chg_pct, vardummy], axis=1)


def fetch_f1r_oo(begin_date, end_date, universe):
    remain_data = RetrievalAPI.get_data_by_map(
        begin_date=begin_date,
        end_date=end_date,
        columns=['ret_f1r_oo', 'dummy_test_f1r_open', universe],
        method='ddb')
    return remain_data
