import pdb
import os
from dotenv import load_dotenv
load_dotenv()
from lib.utils.ttimes import get_dates
from lib.utils.macro import base_path
from lib.data.process_data import process_tick_data

# 处理tick主力合约
def start1(method):
    output_path = os.path.join(base_path, 'data', 'main_tick')
    os.makedirs(output_path, exist_ok=True)
    start_date, end_date = get_dates(method)
    process_tick_data(base_path=os.environ['TICK_FUT_DIRS'],
                      begin_date=start_date, end_date=end_date,
                      codes=['HC'],
                      output_path=output_path)


if __name__ == '__main__':
    start1(method='ricso2')
