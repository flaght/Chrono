import pdb, os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro import base_path
from kdutils.ttimes import get_dates
from alphacopilot.api.data import DDBAPI, ddb_tools
from alphacopilot.calendars.api import advanceDateByCalendar


def fetch_spot_daily(begin_date, end_date, code):
    cusomize_api = DDBAPI.cusomize_api()
    clause_list1 = ddb_tools.to_format('Code', 'in', [code])
    clause_list2 = ddb_tools.to_format(
        'date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))

    clause_list3 = ddb_tools.to_format(
        'date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
    results = cusomize_api.custom(
        table='index_market',
        columns=[
            'date', 'Code', 'openIndex', 'highestIndex', 'lowestIndex',
            'closeIndex', 'turnoverVol', 'turnoverValue'
        ],
        clause_list=[clause_list1, clause_list2, clause_list3],
        format_data=1,
        db_path='tl_daily')
    results.rename(columns={
        'date': 'trade_date',
        'Code': 'code',
        'openIndex': 'open',
        'highestIndex': 'high',
        'lowestIndex': 'low',
        'closeIndex': 'close',
        'turnoverVol': 'volume',
        'turnoverValue': 'value'
    },
                   inplace=True)
    return results


def load_data(begin_date, end_date, code, max_window):
    start_date = advanceDateByCalendar('china.sse', begin_date, '-4b')
    end_date1 = advanceDateByCalendar('china.sse', end_date,
                                      '{0}b'.format(max_window))
    market_data = fetch_spot_daily(begin_date=start_date.strftime('%Y-%m-%d'),
                                   end_date=end_date1.strftime('%Y-%m-%d'),
                                   code=code)
    return market_data.sort_values(by=['trade_date'])


def create_chg(market_data, name='vwap'):
    pricep = market_data.set_index(['trade_date', 'code'])[name].unstack()
    pre_pricep = pricep.shift(1)
    ret_v2v = np.log((pricep) / pre_pricep)
    yields_data = ret_v2v.shift(-2)
    yields_data = yields_data.stack()
    yields_data.name = 'chg_pct'
    return yields_data.reset_index()


def create_yields(data, horizon, offset=0):
    df = data.copy()
    df.set_index("trade_date", inplace=True)
    ## chg为log收益
    df['nxt1_ret'] = df['chg_pct']
    df = df.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    df.name = 'nxt1_ret'
    return df


def create_returns(method):
    code = '000852'
    horizon_sets = [1, 2, 3, 5, 10, 15]
    begin_date, end_date = get_dates(method=method)
    market_data = load_data(begin_date=begin_date,
                            end_date=end_date,
                            code=code,
                            max_window=horizon_sets[-1]+10)

    chg_data = create_chg(market_data=market_data, name='close')
    res = []
    for horizon in horizon_sets:
        df = create_yields(data=chg_data.copy(), horizon=horizon)
        df.name = "nxt1_ret_{0}h".format(horizon)
        res.append(df)

    return_data = pd.concat(res, axis=1).reset_index()
    return_data = return_data[(return_data['trade_date'] >= begin_date)
                              & (return_data['trade_date'] <= end_date)]
    base_dir = os.path.join("records", "basic", method)
    os.makedirs(base_dir, exist_ok=True)
    return_data.reset_index(drop=True).to_feather(
        os.path.join(base_dir, "return_data.feather"))


if __name__ == '__main__':
    method = 'train0'
    create_returns(method=method)
