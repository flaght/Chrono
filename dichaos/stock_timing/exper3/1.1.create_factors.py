import os, pdb
from kdutils.ttime import get_dates
from kdutils.fetch_data import fetch_main_daily
from kdutils.yields import fetch_returns
from dotenv import load_dotenv

load_dotenv()

os.environ['INSTRUMENTS'] = 'ifs'
g_instruments = os.environ['INSTRUMENTS']


def fetch_technical(start_date, end_date, method='aicso2'):
    market_data = fetch_main_daily(start_date, end_date, codes=['IF'])
    returns_data = fetch_returns(market_data)
    dirs = os.path.join(os.environ['BASE_PATH'], 'data', method, 'technical')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    pdb.set_trace()
    filename = os.path.join(dirs, 'market_data.feather')
    market_data.to_feather(filename)

    filename = os.path.join(dirs, 'returns_data.feather')
    returns_data.to_feather(filename)


def main(method):
    start_date, end_date = get_dates(method)
    fetch_technical(start_date, end_date, method)


main('bsotr1')
