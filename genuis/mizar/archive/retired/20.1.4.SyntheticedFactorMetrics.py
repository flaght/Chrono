import os, pdb, json
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()

from lib.cux001 import FactorEvaluate1
from kdutils.macro2 import *


def run(method, instruments, period, task_id, name, form):
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        task_id, str(period))

    filename = os.path.join(dirs, "{0}.feather".format(form, name))
    predict_data = pd.read_feather(filename)

    #is_on_mark = predict_data['trade_time'].dt.minute % int(period) == 0
    #predict_data = predict_data[is_on_mark]
    predict_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    predict_data.dropna(inplace=True)

    evaluate1 = FactorEvaluate1(factor_data=predict_data,
                                factor_name='predict',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=15,
                                fee=0.000,
                                scale_method='roll_zscore',
                                resampling_win=15,
                                expression="{0}_{1}".format(form, name))
    pdb.set_trace()
    stats_dt1 = evaluate1.run()
    print(json.dumps(stats_dt1, indent=4, ensure_ascii=False))

    stats_dt2 = evaluate1.run()
    print(json.dumps(stats_dt2, indent=4, ensure_ascii=False))
    


if __name__ == '__main__':
    method = 'cicso0'
    instruments = 'ims'
    period = 15
    name = 'final'
    task_id = '200037'
    form = 'linear_nav_1_90_data'
    run(method=method,
        instruments=instruments,
        period=period,
        task_id=task_id,
        name=name,
        form=form)
