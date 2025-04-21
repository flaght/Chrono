import os, sys, datetime, pdb
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath('../'))

from plugin.chaos.signalor import Signalor


def create_time(begin_time, end_time, freq='1T'):
    """
    Create a time range with the specified frequency.
    """
    return pd.date_range(start=begin_time, end=end_time, freq=freq)


def main():
    #symbol = 'rb2510'
    code = 'RB'
    trade_time = pd.to_datetime('2025-04-21 15:00:00')
    begin_time = pd.to_datetime('2025-04-18 22:45:00')
    trade_times = create_time(begin_time, trade_time, freq='1T')
    actuator = Signalor(code=code)
    for trade_time in trade_times:
        actuator.run(trade_time=pd.to_datetime(trade_time.strftime('%Y-%m-%d %H:%M:%S')))


main()
