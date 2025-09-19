import pdb
import pandas as pd
from alphacopilot.api.data import RetrievalAPI, DDBAPI, ddb_tools
from alphacopilot.api.calendars import advanceDateByCalendar


def fetch_main_daily(begin_date, end_date, codes, columns):
    rename_map = {
        'openPrice': 'open',
        'highestPrice': 'high',
        'lowestPrice': 'low',
        'closePrice': 'close',
        'turnoverVol': 'volume',
        'turnoverValue': 'amount',
        'turnoverRate': 'turnover'
    }
    data = RetrievalAPI.get_data_by_map(columns=columns,
                                        codes=codes,
                                        begin_date=begin_date,
                                        end_date=end_date,
                                        is_debug=1)
    data = {rename_map.get(k, k): v for k, v in data.items()}
    return data


def fetch_specify_data(begin_date, end_date, codes, columns):
    data = RetrievalAPI.get_data_by_map(columns=columns,
                                        codes=codes,
                                        begin_date=begin_date,
                                        end_date=end_date,
                                        is_debug=1)
    return data


def fetch_moneyflow_data(begin_date, end_date, codes):

    clause1 = ddb_tools.to_format("trade_date", ">=",
                                  ddb_tools.convert_date(begin_date))
    clause2 = ddb_tools.to_format("trade_date", "<=",
                                  ddb_tools.convert_date(end_date))
    clause3 = ddb_tools.to_format("code", "in", codes)
    cusomize_api = DDBAPI.cusomize_api()
    moneyflow_data = cusomize_api.custom(
        table='market_moneyflow',
        columns=[
            'trade_date', 'code', 'turnoverValue', 'turnoverVol', 'dealAmount',
            'netFlow', 'inflow', 'outflow', 'mainFlow', 'smainFlow',
            'mainInflow', 'mainOutflow', 'netFlowS', 'netFlowM', 'netFlowL',
            'netFlowXL', 'inflowS', 'inflowM', 'inflowL', 'inflowXL',
            'outflowS', 'outflowM', 'outflowL', 'outflowXL', 'netVol',
            'buyVol', 'sellVol', 'mainBuyVol', 'mainSellVol', 'buyVolS',
            'buyVolM', 'buyVolL', 'buyVolXL', 'sellVolS', 'sellVolM',
            'sellVolL', 'sellVolXL', 'netOrd', 'buyOrd', 'sellOrd',
            'mainBuyOrd', 'mainSellOrd', 'buyOrdS', 'buyOrdM', 'buyOrdL',
            'buyOrdXL', 'sellOrdS', 'sellOrdM', 'sellOrdL', 'sellOrdXL',
            'netFlowRate', 'inflowRate', 'outflowRate', 'mainFlowRate',
            'smainFlowRate', 'mainInflowRate', 'mainOutflowRate',
            'netFlowSRate', 'netFlowMRate', 'netFlowLRate', 'netFlowXLRate',
            'inflowSRate', 'inflowMRate', 'inflowLRate', 'inflowXLRate',
            'outflowSRate', 'outflowMRate', 'outflowLRate', 'outflowXLRate',
            'net_in_cls', 'net_in_opn', 'netVolRate', 'buyVolRate',
            'sellVolRate', 'mainBuyVolRate', 'mainSellVolRate', 'netOrdRate',
            'buyOrdRate', 'sellOrdRate', 'mainBuyOrdRate', 'mainSellOrdRate'
        ],
        clause_list=[clause1, clause2, clause3])

    return moneyflow_data


def fetch_clouto_data(begin_date, end_date, codes):
    cusomize_api = DDBAPI.cusomize_api()
    clause1 = ddb_tools.to_format("statisticsDate", ">=",
                                  ddb_tools.convert_date(begin_date))
    clause2 = ddb_tools.to_format("statisticsDate", "<=",
                                  ddb_tools.convert_date(end_date))
    clause3 = ddb_tools.to_format("code", "in", codes)
    pools = ['social_xueqiu', 'social_guba']
    res1 = []
    for pool in pools:
        df1 = cusomize_api.custom(
            table=pool,
            columns=['code', 'statisticsDate', 'postNum'],
            clause_list=[clause1, clause2, clause3])
        if pool == 'social_xueqiu':
            df1 = df1.rename(
                {
                    'statisticsDate': 'trade_date',
                    'postNum': 'xueqiu'
                }, axis=1)
        elif pool == 'social_guba':
            df1 = df1.rename(
                {
                    'statisticsDate': 'trade_date',
                    'postNum': 'guba'
                }, axis=1)

        res1.append(df1.set_index(['trade_date', 'code']))
    total_data = pd.concat(res1, axis=1, join='outer')
    total_data['guba'] = total_data['guba'].fillna(0)
    total_data['xueqiu'] = total_data['xueqiu'].fillna(0)
    return total_data.unstack()


def fetch_hotmoney(begin_date, end_date, codes):
    ### 先读取成交量
    cusomize_api = DDBAPI.cusomize_api()
    specify_data = fetch_specify_data(begin_date=begin_date,
                                      end_date=end_date,
                                      codes=codes,
                                      columns=['turnoverVol', 'turnoverValue'])
    volume_data = specify_data['turnoverVol'].stack()
    volume_data.name = 'volume'
    volume_data = volume_data.reset_index()

    value_data = specify_data['turnoverValue'].stack()
    value_data.name = 'value'
    value_data = value_data.reset_index()

    clause1 = ddb_tools.to_format("trade_date", ">=",
                                  ddb_tools.convert_date(begin_date))
    clause2 = ddb_tools.to_format("trade_date", "<=",
                                  ddb_tools.convert_date(end_date))
    clause3 = ddb_tools.to_format("code", "in", codes)

    stocks_data = cusomize_api.custom(table='rank_list_stocks',
                                      columns=[
                                          'trade_date', 'code', 'deviation',
                                          'turnoverVol', 'turnoverValue'
                                      ],
                                      clause_list=[clause1, clause2, clause3])
    stocks_data = stocks_data.rename(columns={
        'turnoverVol': 'hotvol',
        'turnoverValue': 'hotvalue'
    })
    stocks_data['on_list'] = 1  # 用户统计上榜次数

    clause1 = ddb_tools.to_format("trade_date", ">=",
                                  ddb_tools.convert_date(begin_date))
    clause2 = ddb_tools.to_format("trade_date", "<=",
                                  ddb_tools.convert_date(end_date))
    clause3 = ddb_tools.to_format("code", "in", codes)

    sales_data = cusomize_api.custom(
        table='rank_list_sales',
        columns=['trade_date', 'code', 'buyValue', 'sellValue'],
        clause_list=[clause1, clause2, clause3])
    sales_data = sales_data.rename(columns={
        'buyValue': 'buy',
        'sellValue': 'sell'
    })
    sales_data1 = sales_data.groupby(['trade_date',
                                      'code']).sum().reset_index()
    hotmoney_data = volume_data.merge(
        value_data, on=['trade_date', 'code'],
        how='left').merge(stocks_data, on=['trade_date', 'code'],
                          how='left').merge(sales_data1,
                                            on=['trade_date', 'code'],
                                            how='left')
    return hotmoney_data.set_index(['trade_date', 'code']).unstack()
