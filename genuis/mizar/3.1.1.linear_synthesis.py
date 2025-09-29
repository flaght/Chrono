import pandas as pd
import numpy as np
import pdb, argparse
import os, pdb, math, itertools
from dotenv import load_dotenv

load_dotenv()
from ultron.factor.genetic.geneticist.operators import *
from lib.lsx001 import fetch_data1, create_factors, create_equal, fetch_chosen_factors


def run(method, instruments, period, datasets=['train', 'val', 'test']):
    expressions = fetch_chosen_factors(method=method, instruments=instruments)
    total_data = fetch_data1(method=method,
                             instruments=instruments,
                             datasets=datasets,
                             period=period,
                             expressions=expressions)
    factors_data = create_factors(total_data=total_data,
                                  expressions=expressions)
    final_data = create_equal(factors_data=factors_data,
                              total_data=total_data,
                              period=period)
    return final_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--method',
                        type=str,
                        default='aicso0',
                        help='data method')
    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instruments')

    parser.add_argument('--period', type=int, default=5, help='period')

    args = parser.parse_args()
    run(method=args.method, instruments=args.instruments, period=args.period)
