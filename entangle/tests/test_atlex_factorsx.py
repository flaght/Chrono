import os, sys, datetime, pdb
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath('../'))

from plugin.atlex.factorx import Factorx



def main():
    symbol = 'rb2510'
    trade_time = pd.to_datetime('2025-04-21 15:00:00')
    factorx = Factorx(symbol=symbol, n_job=4)
    factorx.impluse_run(trade_time=trade_time)



main()
