import datetime, pdb
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

from lumina.formual.impulse import Impulse
from lumina.formual.iactuator import Iactuator

# from ultron.sentry.api import *
# from alphacopilot.api.calendars import advanceDateByCalendar
# from kdutils.data import fetch_main_market, fetch_trader_market1
from lib.attr001.logic001 import fetch_market_data

from config.contract import INSTRUMENTS_CODES


### bar 行情数据对比
def start1(instruments, tick_size):
    adjusted_method = None
    price_fields = ['open', 'high', 'low', 'close', 'vwap']
    rel_fiedls = ["volume", "value", "openint"]
    cover_cols = ["volume", "value", "vwap"]

    begin_time = datetime.datetime(2026, 5, 6)
    end_time = datetime.datetime(2026, 5, 13)

    research_market, trader_market, metrics_data = fetch_market_data(
        instruments=instruments,
        begin_time=begin_time,
        end_time=end_time,
        tick_size=tick_size,
        adjusted_method=adjusted_method,
        price_fields=price_fields,
        rel_fiedls=rel_fiedls,
        cover_cols=cover_cols)
    
    print(metrics_data['results']["field_status"])

    # research_market = fetch_research_data(instruments=instruments,
    #                                       begin_time=begin_time,
    #                                       end_time=end_time,
    #                                       adjusted_method=adjusted_method)

    # trader_market = fetch_trader_data(instruments=instruments,
    #                                   begin_time=begin_time,
    #                                   end_time=end_time,
    #                                   adjusted_method=adjusted_method)

    # research_market, trader_market = algin_data2(research_market,
    #                                              trader_market)

    # ## 之前数据弄错了， 暂时设置为一样。目前已经修改和文华财经一致
    # ## 成交量计算错误
    # for col in cover_cols:
    #     trader_market[col] = research_market[col]

    # ## 价格差异指标
    # price_metrics = price_diff_metrics(research_market=research_market,
    #                                    trader_market=trader_market,
    #                                    tick_size=tick_size,
    #                                    price_fields=price_fields)

    # rel_metrics = relative_diff_metrics(research_market=research_market,
    #                                     trader_market=trader_market,
    #                                     rel_fields=rel_fiedls)
    # price_metrics = pd.DataFrame(price_metrics)
    # rel_metrics = pd.DataFrame(rel_metrics)
    # results = generate_bar_status(price_metrics, rel_metrics)
    # print(results["overall_status"])
    # print(results["field_status"])


if __name__ == '__main__':
    start1(instruments='rbb', tick_size=1)
