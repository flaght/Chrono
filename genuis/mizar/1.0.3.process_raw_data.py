import datetime, pdb, re, os, time
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from dotenv import load_dotenv

load_dotenv()

from lib.agg001.builder import minute_bars
#from lib.agg001.builder2 import minute_bars
from kdutils.data import fetch_basic1
from alphacopilot.calendars.api import makeSchedule, BizDayConventions, DateGeneration

# TICK_DIRS = "/workspace/data/fut_tick/7050707549_-/"
# OUTPUT_DIRS = "/workspace/data/dev/kd/intelkit/records/raw_data/cn_futures"

pattern = re.compile(r'^([A-Za-z]+[0-9]+)_(\d{8})\.csv$')


def parallel_minute_bars(symbol, name, file_path, contMultNum,
                         night_session_start, exchangeCD, output_dirs):
    filename = os.path.join(output_dirs, name).replace(".csv", ".feather")
    time1 = time.time()
    print("symbol poll:{0}".format(symbol, time.time() - time1))
    dt = minute_bars(csv_path=file_path,
                     multiplier=contMultNum,
                     night_session_start=night_session_start,
                     exchange=exchangeCD)
    # print("symbol poll:{0}".format(symbol, time.time() - time1))
    if not dt.empty:
        dt.reset_index(drop=True).to_feather(filename)


def run1(trade_date):
    results = []
    # folder_path = Path("{0}/{1}/{2}".format(os.environ['TICK_FUT_DIRS'],
    #                                         trade_date.strftime("%Y%m"),
    #                                         trade_date.strftime("%Y%m%d")))

    # folder_path = Path("{0}/{1}/{2}/{3}/{4}".format(os.environ['TICK_FUT_DIRS'],
    #                                         trade_date.strftime("%Y"),
    #                                         trade_date.strftime("%Y%m"),
    #                                         trade_date.strftime("%Y%m"),
    #                                         trade_date.strftime("%Y%m%d")))
    
    folder_path = Path("{0}/{1}/{2}/{3}".format(os.environ['TICK_FUT_DIRS'],
                                            trade_date.strftime("%Y"),
                                            trade_date.strftime("%Y%m"),
                                            trade_date.strftime("%Y%m%d")))
    
    for csv_file in folder_path.glob("*.csv"):
        print(csv_file)
        filename = csv_file.name
        match = pattern.match(filename)
        if match:
            contract = match.group(1)  # 比如 'IH2606'
            date_str = match.group(2)  # 比如 '20260521'

            results.append({
                'file_path': csv_file,
                'symbol': contract,
                'date': date_str,
                'name': filename
            })
        else:
            print(f"⚠️ 跳过不合规或乱码文件: {filename}")
    #pdb.set_trace()
    symbols = [result['symbol'] for result in results]
    basic_info = fetch_basic1(begin_date=trade_date,
                              end_date=trade_date,
                              symbols=symbols)
    results = pd.DataFrame(results)
    # pdb.set_trace()
    basic_info = results.merge(basic_info, on=['symbol'])
    output_dirs = os.path.join(os.environ['BAR_FUT_DIRS'],
                               trade_date.strftime("%Y%m%d"))
    os.makedirs(output_dirs, exist_ok=True)
    _ = Parallel(n_jobs=128, verbose=1)(delayed(parallel_minute_bars)(
        symbol=row.symbol,
        name=row.name,
        file_path=row.file_path,
        contMultNum=row.contMultNum,
        night_session_start=datetime.time(20, 0, 0),
        exchangeCD=row.exchangeCD,
        output_dirs=output_dirs) for row in basic_info.itertuples())

    # for row in basic_info.itertuples():
    #     dt = minute_bars(csv_path=row.file_path,
    #                      multiplier=row.contMultNum,
    #                      night_session_start=datetime.time(20, 0, 0),
    #                      exchange=row.exchangeCD)
    #     filename = os.path.join(output_dirs,
    #                             row.name).replace(".csv", ".feather")
    #     if not dt.empty:
    #         dt.reset_index(drop=True).to_feather(filename)


def start1(start_time, end_time):
    dates = makeSchedule(start_time,
                         end_time,
                         '1b',
                         calendar='china.sse',
                         dateRule=BizDayConventions.Following,
                         dateGenerationRule=DateGeneration.Backward)
    for date in dates:
        run1(trade_date=date)


if __name__ == '__main__':
    start1(start_time='2026-06-30', end_time='2026-06-30')
