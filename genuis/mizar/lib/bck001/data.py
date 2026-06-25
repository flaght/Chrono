from lib.attr001.ftd001 import *

def load_market_data(instruments, begin_time, end_time, trading_sessions):
    market_data = fetch_research_data(instruments=instruments,
                                      begin_time=begin_time,
                                      end_time=end_time,
                                      adjusted_method=None)

    market_data = filter_trading_time(data=market_data,
                                      trading_sessions=trading_sessions)
    return market_data
