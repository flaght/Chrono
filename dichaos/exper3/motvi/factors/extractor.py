import pdb
import pandas as pd
from alphacopilot.api.data import RetrievalAPI, DDBAPI, ddb_tools
from alphacopilot.api.calendars import advanceDateByCalendar


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
    market_data = market_data.drop(['symbol'], axis=1)
    returns_data = market_data.set_index(
        ['trade_date', 'code'])['close'].unstack().pct_change().stack()
    returns_data = returns_data.loc[begin_date:end_date]
    returns_data.name = 'returns'
    market_data = pd.concat([market_data.set_index(['trade_date', 'code']), returns_data], axis=1)
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
