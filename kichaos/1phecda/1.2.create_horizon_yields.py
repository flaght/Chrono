import pdb, os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ['INSTRUMENTS'] = 'ims'
g_instruments = os.environ['INSTRUMENTS']

from alphacopilot.api.calendars import advanceDateByCalendar
from kdutils.macro import base_path, codes
from kdutils.ttimes import get_dates
from kdutils.data import fetch_main_market


def create_yields(data, horizon, offset=0):
    df = data.copy()
    df.set_index("trade_time", inplace=True)
    ## chg为log收益
    df['nxt1_ret'] = df['chg_pct']
    df = df.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    df.name = 'nxt1_ret'
    return df


def build_yields(method, begin_date, end_date, categories, horizon_sets=[]):
    pdb.set_trace()
    data = fetch_main_market(begin_date=begin_date,
                             end_date=end_date,
                             codes=codes)

    ### 收益率 o2o T+1期开盘价和T+2期开盘价比. T期算因子, T+1期的开盘价交易，T+2期开盘价为一次收益计算
    if categories == 'o2o':
        openp = data.set_index(['trade_time', 'code'])['open'].unstack()
        pre_openp = openp.shift(1)
        ret_o2o = np.log((openp) / pre_openp)
        yields_data = ret_o2o.shift(-2)
        yields_data = yields_data.stack()
        yields_data.name = 'chg_pct'
        yields_data = yields_data.reset_index()
    ##持仓目标收益率
    dirs = os.path.join(base_path, method, g_instruments, 'yields')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    for horizon in horizon_sets:
        df = create_yields(data=yields_data, horizon=horizon)
        filename = os.path.join(dirs,
                                '{0}_{1}h.feather'.format(categories, horizon))
        pdb.set_trace()
        df.reset_index().to_feather(filename)
        print('save yields data to {0}'.format(filename))


def main(method):
    start_date, end_date = get_dates(method)
    start_time = advanceDateByCalendar('china.sse', start_date,
                                       '-{0}b'.format(0)).strftime('%Y-%m-%d')
    build_yields(method, start_time, end_date, 'o2o', [1, 3, 5])


main('aicso3')
