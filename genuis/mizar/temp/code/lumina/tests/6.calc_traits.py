# -*- encoding:utf-8 -*-
import os, sys, pdb
import pandas as pd
import numpy as np

from create_data import load_random_data


sys.path.insert(0, os.path.abspath('../'))

import lumina.traits.base.i001 as i00

def create_data():
    columns = ['close','low','high','open','volume','value','openint','chg', 'price']
    data = load_random_data(ticker_dim=4, factors_dim=len(columns) - 1, res_name=None)
    data = data.set_index(['trade_time', 'code'])
    data.columns = columns
    return data.unstack()



def factors():
    data = create_data()
    i00.Traits001().run(data)
    
def main():
    factors()

main()