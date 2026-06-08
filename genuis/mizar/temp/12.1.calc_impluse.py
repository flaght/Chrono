import os, sys, pdb
import pandas as pd
import numpy as np

from create_data import load_random_data


sys.path.insert(0, os.path.abspath('../'))
#import lumina.impulse.i013 as i00
import lumina.impulse.i017 as i00

def create_data():
    columns = ['close','low','high','open','volume','value','openint','chg', 'price']
    data = load_random_data(ticker_dim=4, factors_dim=len(columns) - 1, res_name=None)
    data = data.set_index(['trade_time', 'code'])
    data.columns = columns
    return data.unstack()


def factors():
    pdb.set_trace()
    data = create_data()
    for f in i00.__all__:
        print(f)
        cls = getattr(i00, f)
        obj = cls()
        res = obj.calc_impulse(data.copy())
        values = list(res.values())
        dt = pd.concat(values, axis=1).sort_index()
        print("{0}\n\n".format(dt.tail(15)))

def main():
    factors()


main()
