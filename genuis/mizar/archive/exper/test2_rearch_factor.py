import pdb, itertools, os
import pandas as pd
import multiprocessing as mp
from dotenv import load_dotenv

load_dotenv()

os.environ['IMPLUSE_VERSION'] = 'all'

from kdutils.tactix import Tactix
from lib.iux001 import fetch_data
from config.contract import INSTRUMENTS_CODES, INDEX_MAPPING
from lib.cux001 import FactorEvaluate1

import lumina.impulse.i016 as i000
import lumina.impulse.i017 as i000

_PARALLEL_TOTAL_DATA = None


def parallel_evaluate1(factor_name, instruments, outputs):
    total_data1 = _PARALLEL_TOTAL_DATA
    print(factor_name)
    try:
        evaluate1 = FactorEvaluate1(factor_data=total_data1,
                                factor_name=factor_name,
                                ret_name='nxt1_ret_{0}h'.format(5),
                                roll_win=15,
                                fee=0.0,
                                scale_method='roll_zscore',
                                expression='{0}_{1}'.format(
                                    instruments, factor_name),
                                resampling_win=5,
                                name='{0}_{1}'.format(instruments,
                                                      factor_name))
        stats_dt = evaluate1.run()
        evaluate1.plot_results()
        evaluate1.save_results(base_output_dir=outputs)
    except Exception as e:
        print("{0}".format(str(e)))


def start(method, instruments, task_id, period):
    total_data = fetch_data(
        method=method,
        instruments=instruments,
        task_id=INDEX_MAPPING[INSTRUMENTS_CODES[instruments]],
        datasets=['recent'])
    data = total_data.set_index(['trade_time', 'code']).unstack()

    for f in i000.__all__:
        #cls = getattr(i000, "Impulse{0}".format(f))
        print("start:{0}".format(f))
        cls = getattr(i000, f)
        obj = cls()
        r1 = obj.calc_impulse(data.copy())
        values = list(r1.values())
        values1 = [v.sort_index() for v in values]
        dt = pd.concat(values1, axis=1).sort_index()
        features = dt.columns.tolist()
        total_data1 = dt.merge(
            total_data[['trade_time', 'code', 'nxt1_ret_5h']],
            on=['trade_time', 'code'])

        global _PARALLEL_TOTAL_DATA
        _PARALLEL_TOTAL_DATA = total_data1

        outputs = os.path.join("records", "alabo",
                               method, "nxt1_ret_{}h".format(str(period)),
                               str(task_id), f)

        args_for_starmap = zip(features, itertools.repeat(instruments),
                               itertools.repeat(outputs))

        if len(features) <= 1:
            res = [
                parallel_evaluate1(
                    features[0], instruments=instruments, outputs=outputs)
            ] if features else []
        else:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context()
            with ctx.Pool(processes=1) as pool:
                res = pool.starmap(parallel_evaluate1, args_for_starmap)


if __name__ == '__main__':
    #variant = Tactix().start()
    codes = ['rbb','hc']
    for instrument in ['rbb', 'mab', 'vi', 'nii', 'mc']:
        start(method='ricso2',
              instruments=instrument,
              task_id='22222',
              period=5)
