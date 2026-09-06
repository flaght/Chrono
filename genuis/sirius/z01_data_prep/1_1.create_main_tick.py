import pdb
import os
from dotenv import load_dotenv
load_dotenv()
from lib.utils.macro import base_path
from lib.utils.ttimes import get_dates
from lib.data.load_data import fetch_tick_data


def create_factors(method, instruments):
    start_date, end_date = get_dates(method)
    tick_data = fetch_tick_data(base_path=base_path,
                                begin_date=start_date,
                                end_date=end_date, codes=['RB'])
    output_path = os.path.join(base_path, method, instruments, 'basic')
    os.makedirs(output_path, exist_ok=True)
    tick_data.reset_index(drop=True).to_feather(os.path.join(output_path, 'tick_data.feather'))


if __name__ == '__main__':
    create_factors(method='ricso2', instruments='rbb')
