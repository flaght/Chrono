## Crypto BN 因子计算
import pdb, os, datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import lumina.env as env

env.g_format = 2

import lumina.impulse.i001 as i001
import lumina.impulse.i002 as i002
import lumina.impulse.i003 as i003
import lumina.impulse.i004 as i004
import lumina.impulse.i005 as i005
import lumina.impulse.i006 as i006
import lumina.impulse.i007 as i007
import lumina.impulse.i008 as i008
import lumina.impulse.i009 as i009
import lumina.impulse.i010 as i010
import lumina.impulse.i011 as i011
import lumina.impulse.i012 as i012
import lumina.impulse.i013 as i013
import lumina.impulse.i014 as i014

from kdutils.ttimes import get_dates
from kdutils.macro2 import base_path
from kdutils.tactix import Tactix


def callback_save(factors_data, name, task_id, method):
    start_date, end_date = get_dates(method)
    # cond1 = (factors_data.index.get_level_values(level=0)
    #          >= (datetime.datetime.strptime(start_date, '%Y-%m-%d') +
    #              datetime.timedelta(days=1)).strftime('%Y-%m-%d')) & (
    #                  factors_data.index.get_level_values(level=0)
    #                  <= (datetime.datetime.strptime(end_date, '%Y-%m-%d') +
    #                      datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    cond1 = (factors_data['trade_time']
             >= (datetime.datetime.strptime(start_date, '%Y-%m-%d') +
                 datetime.timedelta(days=1)).strftime('%Y-%m-%d')) & (
                     factors_data['trade_time']
                     <= (datetime.datetime.strptime(end_date, '%Y-%m-%d') +
                         datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    factors_data = factors_data[cond1]
    ff = factors_data.sort_values(by=['trade_time','code']).reset_index(drop=True)
    ff1 = ff
    dirs = os.path.join(base_path, method, 'derivative', task_id)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs,
                            '{0}_factors.feather'.format(name.split('.')[-1]))
    print(filename)
    ff1.sort_index().reset_index(drop=True).to_feather(filename)


def calculate_factors(data, method, task_id, callback):

    def run(data, i00, callback, method, task_id):
        res = []
        data1 = None
        for f in i00.__all__[0:10]:
            print(f)

            cls = getattr(i00, f)
            obj = cls()
            r1 = obj.calc_impulse(data.copy())

            longest_key = max(r1.keys(), key=lambda k: len(r1[k]))
            target_index = r1[longest_key].index
            dt = pd.DataFrame(
                {
                    k: v.reindex(target_index).values
                    for k, v in r1.items()
                },
                index=target_index).sort_index()
            if data1 is None:
                data1 = dt.reset_index()
            else:
                data1 = data1.merge(dt.reset_index(),
                                    on=['trade_time', 'code'])

            # dt = pd.DataFrame(index=data.index).reset_index()
            # for k, v in r1.items():
            #     temp_df = v.rename(k).reset_index()
            #     dt = dt.merge(temp_df, on=['trade_time', 'code'], how='left')
            # values = list(r1.values())
            # values1 = [v.sort_index() for v in values]
            # dt = pd.concat(values1, axis=1).sort_index()
            #res.append(dt.sort_index())
        # data = pd.concat(res, axis=1)
        callback(factors_data=data1,
                 name=i00.__name__,
                 method=method,
                 task_id=task_id)

    for i00 in [
            i001, i002, i003, i004, i005, i006, i007, i008, i009, i010, i011,
            i012, i013, i014
    ]:
        run(data=data,
            i00=i00,
            callback=callback,
            method=method,
            task_id=task_id)


def create_factors(method, task_id):
    dirs = os.path.join(base_path, method, 'basic', task_id)
    file_name = os.path.join(dirs, "raw_basic.feather")
    raw_basic_data = pd.read_feather(file_name)
    pdb.set_trace()
    # raw_basic_data[
    #     's_vwap'] = raw_basic_data['s_value'] / raw_basic_data['s_vol']
    # raw_basic_data[
    #     'f_vwap'] = raw_basic_data['f_value'] / raw_basic_data['f_vol']

    raw_basic_data['vwap'] = raw_basic_data['value'] / raw_basic_data['volume']
    pdb.set_trace()
    raw_basic_data = raw_basic_data.set_index(['trade_time', 'code']).unstack()
    calculate_factors(data=raw_basic_data,
                      method=method,
                      task_id=task_id,
                      callback=callback_save)


if __name__ == '__main__':
    variant = Tactix().start()
    create_factors(method=variant.method, task_id=variant.task_id)
