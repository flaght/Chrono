import os, sys, pdb, re, math, json
import sqlalchemy as sa
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath('../'))
from lumina.genetic import Rotors, Rotor

base_path = '/workspace/worker/pj/Chrono/lumina/abily/records'


def fetch_data():
    positions_data = pd.read_feather(os.path.join(base_path, 'tt.feather'))
    positions_data['trade_time'] = pd.to_datetime(positions_data['trade_time'])
    positions_data = positions_data.set_index('trade_time')
    return positions_data


def train():
    strategy_settings = {
        'capital': 10000000,
        'commission': 0.000023,
        'slippage': 0.0001,
        'size': 300
    }
    filename = os.path.join(base_path, 'aicso2', 'ims', 'merge',
                            "val_data.feather")

    market_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    market_data['trade_time'] = pd.to_datetime(market_data['trade_time'])
    market_data = market_data.set_index(['trade_time', 'code'])[[
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'vwap'
    ]]
    market_data = market_data.unstack()

    rotor = Rotors(signal_values=[-1, 0, 1], k_split=1, n_clusters=3)
    positions_data = fetch_data()

    res = rotor.evaluation(positions_data=positions_data,
                           market_data=market_data,
                           strategy_setting=strategy_settings)

    rotor.save_model(path=os.path.join(base_path, 'aicso2', 'ims', 'kmeans'),
                     best_mapping=res[0].mapping)


def predict():
    path = os.path.join(base_path, 'aicso2', 'ims', 'kmeans','1222222','im')
    rotor = Rotor.from_pickle(path=path)
    positions_data = fetch_data()
    pdb.set_trace()
    current_signals = rotor.predict(positions_data=positions_data)
    print(current_signals)

train()