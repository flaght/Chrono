import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix

from kdutils.macro2 import *
from lib.iux001 import fetch_data
from lib.cux001 import FactorEvaluate1


def run1(method, instruments, period, task_id):
    logic_features = pd.read_csv(
        os.path.join(base_path, "resource", "logic_basic.csv"))
    logic_features = logic_features['name'].unique().tolist()
    total_factors = fetch_data(method=method,
                               task_id=task_id,
                               instruments=instruments,
                               datasets=['train', 'val'])
    total_factors = total_factors
    basic_features = [
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'vwap'
    ]
    f1 = [
        col for col in total_factors.columns
        if col not in ['trade_time', 'code', 'symbol']
    ]
    f1 = f1 + basic_features
    f2 = logic_features
    features = list(set(f1) & set(f2))
    total_factors = total_factors[['trade_time', 'code'] + features +
                                  ['nxt1_ret_{}h'.format(period)]]
    res = []
    for f in features:
        print(f)
        evaluate1 = FactorEvaluate1(factor_data=total_factors,
                                    factor_name=f,
                                    ret_name='nxt1_ret_{0}h'.format(period),
                                    roll_win=15,
                                    fee=0.000,
                                    scale_method='roll_zscore',
                                    resampling_win=period,
                                    expression=f)
        state_dt = evaluate1.run()
        res.append({'name': f, 'autocorr': state_dt['factor_autocorr']})
    features  = pd.DataFrame(res)
    features['abs'] = np.abs(features['autocorr'])
    pdb.set_trace()
    features = features[features['abs'].between(0.2, 0.7)]

if __name__ == '__main__':
    variant = Tactix().start()
    run1(method=variant.method,
         instruments=variant.instruments,
         period=variant.period,
         task_id=variant.task_id)
