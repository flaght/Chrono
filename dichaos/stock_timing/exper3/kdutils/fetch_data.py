import pdb
from alphacopilot.api.data import RetrievalAPI, ddb_tools, DDBAPI
from alphacopilot.dataapi.retireval.customized import *


def fut_daily(begin_date, end_date, symbol, code, columns):
    clause_list1 = to_format('code', 'in', [symbol])
    clause_list2 = to_format('trade_date', '>=', convert_date(begin_date))
    clause_list3 = to_format('trade_date', '<=', convert_date(end_date))
    cusomize_api = DDBAPI.cusomize_api()
    df = cusomize_api.custom(
        table='market_fut',
        columns=columns,
        clause_list=[clause_list1, clause_list2, clause_list3],
        format_data=0)
    df.rename(columns={'code': 'symbol'}, inplace=True)
    df['code'] = code
    return df


def main_daily(begin_date,
               end_date,
               codes,
               method,
               columns=None,
               format_data=1):
    adj_columns = ['closePrice', 'highestPrice', 'lowestPrice', 'openPrice']
    inter_columns = list(set(columns) & set(adj_columns)) if isinstance(
        columns, list) else adj_columns
    adj_name = "{0}_cumfactor".format(method)
    algin_factors = get_algin_factors(
        begin_date=begin_date,
        end_date=end_date,
        codes=codes,
        columns=['trade_date', 'code', 'symbol', adj_name])
    algin_factors = algin_factors.drop_duplicates(
        subset=['trade_date', 'code'], keep='last')
    ex_factors = get_ex_factors(begin_date=begin_date,
                                end_date=end_date,
                                codes=codes,
                                columns=['trade_date', 'end_date', 'code'])
    daily_data = {}
    count = len(ex_factors)
    for index, row in enumerate(ex_factors.itertuples()):
        cond1 = (algin_factors.trade_date >= row.trade_date) & (
            algin_factors.trade_date <= row.end_date) & (algin_factors.code
                                                         == row.code)
        factors = algin_factors[cond1]
        if factors.empty:
            continue
        begin_date = factors.trade_date.min().strftime('%Y-%m-%d')
        end_date = factors.trade_date.max().strftime('%Y-%m-%d')
        end_date = row.end_date.strftime('%Y-%m-%d') if index == (
            count - 1) else end_date
        symbol = factors.symbol.tolist()[-1]
        cumfactors = factors.pcr_cumfactor.tolist()[0]
        md = fut_daily(begin_date, end_date, symbol, row.code, columns)
        md[inter_columns] = (md[inter_columns] * cumfactors)  #.apply(round)
        md_res = []
        if row.code in daily_data:
            md_res = daily_data[row.code]
        md_res.append(md)
        daily_data[row.code] = md_res

    if format_data == 0:
        res = {}
        for code in list(daily_data.keys()):
            res[code] = pd.concat(daily_data[code])
    else:
        res = []
        for code in list(daily_data.keys()):
            res.append(pd.concat(daily_data[code], axis=0))
        res = pd.concat(res, axis=0)
    return res


def fetch_basic(codes):
    clause_list1 = ddb_tools.to_format('flag', '==', 1)
    clause_list2 = ddb_tools.to_format('contractObject', 'in', codes)
    cusomize_api = DDBAPI.cusomize_api()
    basic_data = cusomize_api.custom(
        table='fut_basic',
        clause_list=[clause_list1, clause_list2],
        columns=['code', 'contractObject', 'contMultNum'],
        format_data=0)
    return basic_data.rename(columns={
        'code': 'symbol',
        'contractObject': 'code'
    })


def fetch_main_daily(begin_date, end_date, codes):
    basic_info = fetch_basic(codes)
    data = main_daily(begin_date=begin_date,
                      end_date=end_date,
                      codes=codes,
                      method='pcr',
                      format_data=0,
                      columns=[
                          'trade_date', 'code', 'openPrice', 'highestPrice',
                          'lowestPrice', 'closePrice', 'turnoverVol',
                          'turnoverValue', 'openInt'
                      ])
    res = []
    for code in data.keys():
        dt = data[code]
        dt['trade_date'] = pd.to_datetime(dt['trade_date'])
        dt.rename(columns={
            'closePrice': 'close',
            'lowestPrice': 'low',
            'highestPrice': 'high',
            'openPrice': 'open',
            'turnoverVol': 'volume',
            'turnoverValue': 'value',
            'openInt': 'openint'
        },
                  inplace=True)
        #dt = dt.drop(columns=['symbol', 'trade_date'], axis=1)
        dt['price'] = dt[['high', 'low', 'close', 'open']].mean(axis=1)
        res.append(dt)
    data = pd.concat(res, axis=0)
    ## 临时 过滤重复数据
    #data = data.merge(basic_info, on=['code','symbol'], how='left')
    #data['vwap'] = data['value'] / data['volume'] / data['contMultNum']
    #data = data.dropna(subset=['vwap'])
    data = data.drop_duplicates(subset=['trade_date', 'code']).sort_values(
        by=['trade_date', 'code'])
    return data
