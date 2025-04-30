import os, sys, datetime, pdb
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath('../'))

from plugin.virgtor.signalor import Signalor


def create_time(begin_time, end_time, freq='1T'):
    """
    Create a time range with the specified frequency.
    """
    return pd.date_range(start=begin_time, end=end_time, freq=freq)


def main():
    #symbol = 'rb2510'
    code = 'IM'
    trade_time = pd.to_datetime('2025-04-28 11:30:00')
    begin_time = pd.to_datetime('2025-04-28 09:30:00')
    trade_times1 = create_time(begin_time, trade_time, freq='1T')

    begin_time = pd.to_datetime('2025-04-28 13:30:00')
    trade_time = pd.to_datetime('2025-04-28 15:00:00')
    trade_times2 = create_time(begin_time, trade_time, freq='1T')


    trade_time = pd.to_datetime('2025-04-29 11:30:00')
    begin_time = pd.to_datetime('2025-04-29 09:30:00')
    trade_times3 = create_time(begin_time, trade_time, freq='1T')


    begin_time = pd.to_datetime('2025-04-29 13:30:00')
    trade_time = pd.to_datetime('2025-04-29 15:00:00')
    trade_times4 = create_time(begin_time, trade_time, freq='1T')

    actuator = Signalor(code=code, symbol='IM2506', id='1059861670')
    for trade_time in trade_times1:
        actuator.run(trade_time=pd.to_datetime(
            trade_time.strftime('%Y-%m-%d %H:%M:%S')).strftime('%Y-%m-%d %H:%M:%S'))

    for trade_time in trade_times2:
        actuator.run(trade_time=pd.to_datetime(
            trade_time.strftime('%Y-%m-%d %H:%M:%S')).strftime('%Y-%m-%d %H:%M:%S'))
        
    for trade_time in trade_times3:
        actuator.run(trade_time=pd.to_datetime(
            trade_time.strftime('%Y-%m-%d %H:%M:%S')).strftime('%Y-%m-%d %H:%M:%S'))
        
    for trade_time in trade_times4:
        actuator.run(trade_time=pd.to_datetime(
            trade_time.strftime('%Y-%m-%d %H:%M:%S')).strftime('%Y-%m-%d %H:%M:%S'))
        

main()
