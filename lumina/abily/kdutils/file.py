import os
import pandas as pd

def fetch_file_data(base_path, method, g_instruments, datasets):
    res = []

    def fet(name):
        filename = os.path.join(base_path, method, g_instruments, 'merge',
                                "{0}.feather".format(name))
        factors_data = pd.read_feather(filename).sort_values(
            by=['trade_time', 'code'])
        factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
        return factors_data

    for n in datasets:
        dt = fet(n)
        res.append(dt)
    res = pd.concat(res, axis=0)
    factors_data = res.sort_values(by=['trade_time', 'code'])
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    factors_data = factors_data.sort_values(by=['trade_time', 'code'])
    return factors_data
