### 计算对应参数
import os, pdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from finina.phecda.kdutils.factors1 import factors_sets
from kdutils.macro import base_path, codes
from ultron.kdutils.file import load_pickle
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean


def load_data(method):
    dirs = os.path.join(base_path, method)
    benchmark_kl_pd = load_pickle(
        os.path.join(dirs,
                     'benckmark_{0}.pkl'.format(os.environ['INSTRUMENTS'])))
    pick_kl_pd_dict = load_pickle(
        os.path.join(dirs, 'pick_{0}.pkl'.format(os.environ['INSTRUMENTS'])))
    choice_symbols = load_pickle(
        os.path.join(dirs, 'choice_{0}.pkl'.format(os.environ['INSTRUMENTS'])))
    return benchmark_kl_pd, pick_kl_pd_dict, choice_symbols

def main(method):
    benchmark_kl_pd, pick_kl_pd_dict, choice_symbols = load_data(method)
    pdb.set_trace()
    data = pick_kl_pd_dict["{}0".format(codes[0])]
    ama_line = pd_rolling_mean(data.volume, window=int(5), min_periods=1)
    bma_line = pd_rolling_mean(data.close, window=int(5), min_periods=1)
    bma_chg = bma_line.pct_change(5)
    ama_chg = ama_line.pct_change(5)
    bama = bma_chg * ama_chg
    pdb.set_trace()
    print(bama)
    


main('mini')