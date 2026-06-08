import pandas as pd
import numpy as np
import pdb, argparse
import os, pdb, math, itertools
from dotenv import load_dotenv

load_dotenv()
from ultron.factor.genetic.geneticist.operators import *
from lib.lsx001 import fetch_times, build_factors
from kdutils.macro2 import *


def train_model(method, task_id, instruments, period):
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    filename = os.path.join(dirs, "final_data.feather")
    final_data = pd.read_feather(filename).set_index(['trade_time', 'code'])
    print(final_data.columns)
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

    parser.add_argument('--task_id',
                        type=str,
                        default='200037',
                        help='task id')

    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instruments')

    parser.add_argument('--period', type=int, default=5, help='period')

    args = parser.parse_args()

    build_factors(method=args.method,
                  task_id=args.task_id,
                  instruments=args.instruments,
                  period=args.period)
    train_model(method=args.method,
                task_id=args.task_id,
                instruments=args.instruments,
                period=args.period)
