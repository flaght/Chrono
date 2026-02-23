## Crypto BN 因子计算
import pdb, os, datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import lumina.env as env

env.g_format = 2

import lumina.impulse.c001 as c001

from kdutils.ttimes import get_dates
from kdutils.macro2 import base_path
from kdutils.tactix import Tactix


def callback_save(factors_data, name, period, source, method):
    start_date, end_date = get_dates(method)
    cond1 = (factors_data.index.get_level_values(level=0)
             >= (datetime.datetime.strptime(start_date, '%Y-%m-%d') +
                 datetime.timedelta(days=1)).strftime('%Y-%m-%d')) & (
                     factors_data.index.get_level_values(level=0)
                     <= (datetime.datetime.strptime(end_date, '%Y-%m-%d') +
                         datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    factors_data = factors_data[cond1]
    ff = factors_data.sort_index().reset_index()
    ff1 = ff
    dirs = os.path.join(base_path, method, 'derivative', period, source)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs,
                            '{0}_factors.feather'.format(name.split('.')[-1]))
    print(filename)
    ff1.sort_index().reset_index(drop=True).to_feather(filename)


def calculate_factors(data, method, period, source, callback):

    def run(data, i00, callback, method, period, source):
        res = []
        for f in i00.__all__:
            print(f)
            cls = getattr(i00, f)
            obj = cls()
            r1 = obj.calc_impulse(data.copy())
            values = list(r1.values())
            values1 = [v.sort_index() for v in values]
            dt = pd.concat(values1, axis=1).sort_index()
            res.append(dt)
        data = pd.concat(res, axis=1)
        callback(factors_data=data,
                 name=i00.__name__,
                 method=method,
                 period=period,
                 source=source)

    for i00 in [c001]:
        run(data=data,
            i00=i00,
            callback=callback,
            method=method,
            period=period,
            source=source)


def create_factors(method, period, source):
    dirs = os.path.join(base_path, method, 'basic', period, source)
    file_name = os.path.join(dirs, "raw_basic.feather")
    raw_basic_data = pd.read_feather(file_name)
    raw_basic_data[
        's_vwap'] = raw_basic_data['s_value'] / raw_basic_data['s_vol']
    raw_basic_data[
        'f_vwap'] = raw_basic_data['f_value'] / raw_basic_data['f_vol']
    pdb.set_trace()
    raw_basic_data = raw_basic_data.set_index(['trade_time', 'code']).unstack()
    calculate_factors(data=raw_basic_data,
                      method=method,
                      period=period,
                      source=source,
                      callback=callback_save)


if __name__ == '__main__':
    variant = Tactix().start()
    create_factors(method=variant.method,
                   period=variant.period,
                   source=variant.source)
