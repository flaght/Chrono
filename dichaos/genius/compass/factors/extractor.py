import pdb
import pandas as pd
from alphacopilot.api.data import RetrievalAPI, DDBAPI, ddb_tools
from alphacopilot.api.calendars import advanceDateByCalendar
import os


def fetch_main_returns(begin_date, end_date, codes, columns=None):
    columns = columns if isinstance(
        columns, list) else ['trade_date', 'code', 'closePrice']
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(1))
    market_data = RetrievalAPI.get_main_price(begin_date=start_date,
                                              end_date=end_date,
                                              codes=codes,
                                              method='pcr',
                                              category='daily',
                                              columns=columns)
    market_data.rename(columns={'closePrice': 'close'}, inplace=True)
    market_data = market_data.sort_values(by=['trade_date', 'code', 'close'],
                                          ascending=True).drop_duplicates(
                                              subset=['trade_date', 'code'],
                                              keep='last')
    market_data = market_data.drop(['symbol'], axis=1)
    returns_data = market_data.set_index(
        ['trade_date', 'code'])['close'].unstack().pct_change().stack()
    returns_data = returns_data.loc[begin_date:end_date]
    returns_data.name = 'returns'
    market_data = pd.concat(
        [market_data.set_index(['trade_date', 'code']), returns_data], axis=1)
    return market_data


## 提取主力合约日行情
def fetch_main_daily(begin_date, end_date, codes, columns=None):
    columns = columns if isinstance(columns, list) else [
        'trade_date', 'code', 'openPrice', 'highestPrice', 'lowestPrice',
        'closePrice', 'turnoverVol', 'turnoverValue', 'openInt'
    ]
    market_data = RetrievalAPI.get_main_price(begin_date=begin_date,
                                              end_date=end_date,
                                              codes=codes,
                                              method='pcr',
                                              category='daily',
                                              columns=columns)
    market_data.rename(columns={
        'openPrice': 'open',
        'highestPrice': 'high',
        'lowestPrice': 'low',
        'closePrice': 'close',
        'turnoverVol': 'volume',
        'turnoverValue': 'value',
        'openInt': 'openint'
    },
                       inplace=True)
    market_data = market_data.sort_values(by=['trade_date', 'code', 'openint'],
                                          ascending=True).drop_duplicates(
                                              subset=['trade_date', 'code'],
                                              keep='last')
    market_data = market_data.drop(['symbol'], axis=1)
    market_data = market_data.reset_index(drop=True)
    return market_data


## 提取会员持仓
def fetch_member_positions(begin_date, end_date, codes, category):
    long_positions = RetrievalAPI.get_member_positions(
        begin_date=begin_date,
        end_date=end_date,
        codes=codes,
        category='long',
        columns=['trade_date', 'code', 'partyShortName', 'longVol'])

    short_positions = RetrievalAPI.get_member_positions(
        begin_date=begin_date,
        end_date=end_date,
        codes=codes,
        category='short',
        columns=['trade_date', 'code', 'partyShortName', 'shortVol'])

    if category == 'weighted':
        long_positions['totalLong'] = long_positions.groupby(
            ['trade_date', 'code'])['longVol'].transform('sum')
        long_positions['longVol'] = long_positions['longVol'] * (
            1 + long_positions['longVol'] / long_positions['totalLong'])

        short_positions['totalShort'] = short_positions.groupby(
            ['trade_date', 'code'])['shortVol'].transform('sum')
        short_positions['shortVol'] = short_positions['shortVol'] * (
            1 + short_positions['shortVol'] / short_positions['totalShort'])
    long_positions = long_positions[['trade_date', 'code', 'longVol'
                                     ]].groupby(['trade_date', 'code']).sum()
    short_positions = short_positions[['trade_date', 'code', 'shortVol'
                                       ]].groupby(['trade_date',
                                                   'code']).sum()

    positions_data = pd.concat([long_positions, short_positions], axis=1)
    positions_data.rename(columns={
        'longVol': 'long',
        'shortVol': 'short'
    },
                          inplace=True)
    return positions_data


def fetch_moneyflow_data(begin_date, end_date, codes, category):
    market_universe_dict = {
        'IM': 'zz1000',
        'IC': 'zz500',
        'IF': 'hs300',
        'IH': 'sz50',
        'ashare': 'ashare'
    }

    cusomize_api = DDBAPI.cusomize_api()
    clause1 = ddb_tools.to_format("trade_date", ">=",
                                  ddb_tools.convert_date(begin_date))
    clause2 = ddb_tools.to_format("trade_date", "<=",
                                  ddb_tools.convert_date(end_date))

    clause3 = ddb_tools.to_format("code", "in",
                                  [market_universe_dict[category]])

    market_pure_indicator = cusomize_api.custom(
        table='index_market_moneyflow',
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
            'outflowSRate', 'outflowMRate', 'outflowLRate', 'net_in_cls',
            'net_in_opn', 'outflowXLRate', 'netVolRate', 'buyVolRate',
            'sellVolRate', 'mainBuyVolRate', 'mainSellVolRate', 'netOrdRate',
            'buyOrdRate', 'sellOrdRate', 'mainBuyOrdRate', 'mainSellOrdRate'
        ],
        clause_list=[clause1, clause2, clause3])
    market_pure_indicator['code'] = codes
    market_pure_indicator = market_pure_indicator.set_index(
        ['trade_date', 'code'])
    return market_pure_indicator


'''
def fetch_moneyflow_data(begin_date, end_date, codes, category):

    assert category == codes or category == 'ashare'
    clause_begindate = ddb_tools.to_format("trade_date", ">=",
                                           ddb_tools.convert_date(begin_date))
    clause_enddate = ddb_tools.to_format("trade_date", "<=",
                                         ddb_tools.convert_date(end_date))
    cusomize_api = DDBAPI.cusomize_api()
    mf_flow = cusomize_api.custom(
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
        clause_list=[clause_begindate, clause_enddate])

    ### 这里原始数据的code是股票代码
    pure_indicator = [
        x for x in mf_flow.columns
        if 'Rate' not in x and x != 'trade_date' and x != 'code'
    ]
    derivative_indicator = [
        x for x in mf_flow.columns
        if 'Rate' in x and x != 'trade_date' and x != 'code'
    ]

    market_universe_dict = {
        'IM': 'zz1000',
        'IC': 'zz500',
        'IF': 'hs300',
        'IH': 'sz50',
    }

    if category != 'ashare':
        market_df = cusomize_api.custom(
            table='stk_universe',
            columns=['code', 'trade_date', market_universe_dict[codes]],
            clause_list=[clause_begindate, clause_enddate])
        mf_flow = pd.merge(left=mf_flow,
                           right=market_df,
                           on=['trade_date', 'code'],
                           how='left')
        mf_flow.loc[:,
                    pure_indicator] = mf_flow.loc[:, pure_indicator] * mf_flow[
                        market_universe_dict[codes]].values.reshape(-1, 1)

    mf_flow = mf_flow.set_index(['trade_date', 'code'])
    market_pure_indicator = mf_flow.loc[:,
                                        pure_indicator].groupby(level=0).sum()
    market_pure_indicator['code'] = codes
    for col in derivative_indicator:
        pure_col = col.replace('Rate', '')
        if 'flow' in col or 'Flow' in col:
            need_col = 'turnoverValue'
        elif 'Vol' in col:
            need_col = 'turnoverVol'
        elif 'Ord' in col:
            need_col = 'dealAmount'
        else:
            continue

        market_pure_indicator[col] = market_pure_indicator[
            pure_col] / market_pure_indicator[need_col]
    market_pure_indicator = market_pure_indicator.reset_index().set_index(
        ['trade_date', 'code'])
    return market_pure_indicator
'''


def fetch_heat_data(begin_date, end_date, codes):
    market_universe_dict = {
        'IM': 'zz1000',
        'IC': 'zz500',
        'IF': 'hs300',
        'IH': 'sz50'
    }
    clause1 = ddb_tools.to_format("trade_date", ">=",
                                  ddb_tools.convert_date(begin_date))
    clause2 = ddb_tools.to_format("trade_date", "<=",
                                  ddb_tools.convert_date(end_date))
    clause3 = ddb_tools.to_format("code", "in", [market_universe_dict[codes]])
    cusomize_api = DDBAPI.cusomize_api()
    heat_df = cusomize_api.custom(
        table='index_market_clouto',
        columns=['trade_date', 'code', 'xueqiu', "guba"],
        clause_list=[clause1, clause2, clause3])

    heat_df = heat_df.rename(columns={
        'xueqiu_postNum': 'xueqiu',
        'guba_postNum': 'guba'
    })
    heat_df['code'] = codes
    return heat_df.set_index(['trade_date', 'code']).unstack()


'''
def fetch_heat_data(begin_date, end_date, codes):
    clause_begindate = ddb_tools.to_format("trade_date", ">=",
                                           ddb_tools.convert_date(begin_date))
    clause_enddate = ddb_tools.to_format("trade_date", "<=",
                                         ddb_tools.convert_date(end_date))
    cusomize_api = DDBAPI.cusomize_api()
    market_universe_dict = {
        'IM': 'zz1000',
        'IC': 'zz500',
        'IF': 'hs300',
        'IH': 'sz50',
    }
    # 加载指定股票池
    market_df = cusomize_api.custom(
        table='stk_universe',
        columns=['code', 'trade_date', market_universe_dict[codes]],
        clause_list=[clause_begindate, clause_enddate])

    clause_begindate2 = ddb_tools.to_format("statisticsDate", ">=",
                                            ddb_tools.convert_date(begin_date))
    clause_enddate2 = ddb_tools.to_format("statisticsDate", "<=",
                                          ddb_tools.convert_date(end_date))
    xueqiu_df = cusomize_api.custom(
        table='social_xueqiu',
        columns=['code', 'statisticsDate', 'postNum'],
        clause_list=[clause_begindate2, clause_enddate2])
    guba_df = cusomize_api.custom(
        table='social_guba',
        columns=['code', 'statisticsDate', 'postNum'],
        clause_list=[clause_begindate2, clause_enddate2])

    xueqiu_df = xueqiu_df.rename(
        {
            'statisticsDate': 'trade_date',
            'postNum': 'xueqiu_postNum'
        }, axis=1)
    guba_df = guba_df.rename(
        {
            'statisticsDate': 'trade_date',
            'postNum': 'guba_postNum'
        }, axis=1)

    market_df = market_df.merge(xueqiu_df,
                                on=['trade_date', 'code'],
                                how='left')
    market_df = market_df.merge(guba_df, on=['trade_date', 'code'], how='left')

    market_df['xueqiu_postNum'] = market_df['xueqiu_postNum'].fillna(0)
    market_df['guba_postNum'] = market_df['guba_postNum'].fillna(0)

    market_df['xueqiu_postNum'] = market_df['xueqiu_postNum'] * market_df[
        market_universe_dict[codes]]
    market_df['guba_postNum'] = market_df['guba_postNum'] * market_df[
        market_universe_dict[codes]]
    ### 注意 这里的code为股票代码 并不是择时标的，后续会进行修改填充
    market_df = market_df.set_index(['trade_date', 'code'])
    heat_df = market_df.groupby(
        'trade_date').sum().loc[:, ['xueqiu_postNum', 'guba_postNum']]
    heat_df = heat_df.reset_index()
    heat_df['code'] = codes
    heat_df = heat_df.set_index(['trade_date', 'code'])
    heat_df = heat_df.rename(
        {
            'xueqiu_postNum': 'xueqiu',
            'guba_postNum': 'guba'
        }, axis=1)
    heat_df = heat_df.unstack()
    return heat_df
'''
