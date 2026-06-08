import pdb
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.api.calendars import advanceDateByCalendar

from kdutils.data import fetch_main_market
from kdutils.common import fetch_temp_data
from config.contract import INSTRUMENTS_CODES
from kdutils.ttimes import get_dates



def load_data_from_feather(instruments, method, rootid=0):

    total_factors = fetch_temp_data(method=method,
                                    task_id=rootid,
                                    instruments=instruments,
                                    datasets=['train', 'val', 'test'])
    total_factors = total_factors.set_index(['trade_time', 'code'])

    res = {}
    cols = [
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'chg',
        'price', 'vwap'
    ]
    for col in cols:
        if col in total_factors.columns:
            res[col] = total_factors[col].unstack()
    return res


def load_data_from_dolphin(instruments, start_date, end_date):
    pdb.set_trace()
    market_data = fetch_main_market(begin_date=start_date,
                                    end_date=end_date,
                                    codes=[INSTRUMENTS_CODES[instruments]],
                                    keep_symbol=True)
    market_data = market_data.set_index(['trade_time', 'code'])

    res = {}
    cols = [
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'chg',
        'price', 'vwap'
    ]
    for col in cols:
        res[col] = market_data[col].unstack()
    return res


def run1(instruments, method, task_id):
    start_date, end_date = get_dates(method)

    begin_date = advanceDateByCalendar('china.sse', start_date,
                                       '-{0}b'.format(2)).strftime('%Y-%m-%d')

    pdb.set_trace()
    res1 = load_data_from_dolphin(instruments=instruments,
                                 start_date='2025-06-25',
                                 end_date=end_date)
    pdb.set_trace()
    res2 = load_data_from_feather(instruments=instruments,
                                 method=method,
                                 rootid=task_id)
    
    
    intersection1 = None
    intersection2 = None
    for key1, value1 in res1.items():
        if intersection1 is None:
            intersection1 = value1.dropna().index
        else:
            intersection1 = intersection1.intersection(value1.dropna().index)
            
    for key2, value2 in res2.items():
        if intersection2 is None:
            intersection2 = value2.index
        else:
            intersection2 = intersection2.intersection(value2.index)
            
    intersection = intersection2.intersection(intersection1)
    
    pdb.set_trace()
    for key in res1:
        print(key)
        res1[key] = res1[key].loc[intersection]
    
    for key in res2:
        print(key)
        res2[key] = res2[key].loc[intersection]
        
    pdb.set_trace()
    print()
        
run1(instruments='rbb', method="bicso2", task_id='113001')