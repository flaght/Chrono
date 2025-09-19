import datetime, pdb, os, sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ['INSTRUMENTS'] = 'ifs'
g_instruments = os.environ['INSTRUMENTS']

from kdutils.macro import *
from ultron.factor.genetic.geneticist.operators import calc_factor


def fetch_data(method):
    filename = os.path.join(base_path, method, g_instruments, 'merge',
                            "train_data.feather")
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    factors_data = factors_data.sort_values(by=['trade_time', 'code'])
    factors_data = factors_data.set_index('trade_time')
    return factors_data


method = "aicso1"
total_data = fetch_data(method)

expression = "MSUM(10,MPWMA(20,'rv007_5_10_0_1','ixy012_10_15_1'))"

pdb.set_trace()
factors_data = calc_factor(expression=expression,
                           total_data=total_data,
                           key='code',
                           indexs=[])
factors_data = factors_data.replace([np.inf, -np.inf], np.nan)
factors_data['transformed'] = np.where(np.abs(factors_data.transformed.values) > 0.000001,factors_data.transformed.values, np.nan)
coverage_rate = 1 - factors_data['transformed'].isna().sum() / len(factors_data['transformed'])

print("==>")
