import pandas as pd
import numpy as np
import pdb, argparse
import os, pdb, math, itertools
from dotenv import load_dotenv

load_dotenv()
from ultron.factor.genetic.geneticist.operators import *
from lib.lsx001 import fetch_data1, create_factors, create_equal, fetch_chosen_factors, fetch_times
from kdutils.macro2 import *


def build_factors(method,
                  instruments,
                  period,
                  datasets=['train', 'val', 'test']):
    expressions = fetch_chosen_factors(method=method, instruments=instruments)
    total_data = fetch_data1(method=method,
                             instruments=instruments,
                             datasets=datasets,
                             period=period,
                             expressions=expressions)
    factors_data = create_factors(total_data=total_data,
                                  expressions=expressions)
    dirs = os.path.join(base_path, method, instruments, 'temp', "tree",
                        str(period))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs, "final_data.feather")
    final_data = factors_data.reset_index().merge(
        total_data[['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)]],
        on=['trade_time', 'code'])
    final_data.to_feather(filename)


def train_model(method, instruments, period):
    time_array = fetch_times(method=method, instruments=instruments)
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(period))
    filename = os.path.join(dirs, "final_data.feather")
    final_data = pd.read_feather(filename).set_index(['trade_time', 'code'])
    final_data1 = final_data.mean(axis=1)
    final_data1.name = 'predict'
    final_data1 = pd.concat(
        [final_data1, final_data[['nxt1_ret_{0}h'.format(period)]]], axis=1)
    test_data = final_data1.loc[
        time_array['test_time'][0]:time_array['test_time'][1]]
    test_data.reset_index().to_feather(
        os.path.join(dirs, "linear_predict_data.feather"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--method',
                        type=str,
                        default='cicso0',
                        help='data method')
    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instruments')

    parser.add_argument('--period', type=int, default=5, help='period')

    args = parser.parse_args()
    train_model(method=args.method,
                instruments=args.instruments,
                period=args.period)
