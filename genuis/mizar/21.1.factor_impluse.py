import pdb, os
from dotenv import load_dotenv

load_dotenv()

from lumina.impulse.i018.impulse_cpv001 import ImpulseCpv001
from lumina.impulse.i018.impulse_crpv001 import ImpulseCrpv001
from lumina.impulse.i018.impulse_herd001 import ImpulseHerd001
from lumina.impulse.i018.impulse_uret001 import ImpulseUret001
from lumina.impulse.i018.impulse_f_tsm001 import ImpulseFTsm001
from lumina.impulse.i018.impulse_vp_shadow001 import ImpulseVpShadow001
from kdutils.common import fetch_temp_data, fetch_temp_returns
from lib.cux001 import FactorEvaluate1 as FactorEvaluate1


def load_data(method, task_id, instruments):
    desired_columns = [
        'value', 'vwap', 'open', 'high', 'low', 'volume', 'openint', 'close'
    ]
    total_factors = fetch_temp_data(method=method,
                                    task_id=task_id,
                                    instruments=instruments,
                                    datasets=['train', 'val'],
                                    desired_columns=['trade_time', 'code'] +
                                    desired_columns)

    total_returns = fetch_temp_returns(method=method,
                                       instruments=instruments,
                                       datasets=['train', 'val'],
                                       category='returns')

    total_data = total_factors.merge(total_returns,
                                     on=['trade_time', 'code'
                                         ]).set_index(['trade_time', 'code'])
    data_unstack = {}

    for col in desired_columns:
        data_unstack[col] = total_data[col].unstack()

    return total_data, data_unstack


def run(method, instruments, period, task_id):
    market_data, market_unstack = load_data(method=method,
                                            task_id=task_id,
                                            instruments=instruments)
    impulse_cpv = ImpulseVpShadow001()
    class_name = impulse_cpv.name
    factor_unstack = impulse_cpv.calc_impulse(kl_pd=market_unstack)
    for f in factor_unstack.keys():
        dt1 = factor_unstack[f]
        dt1 = dt1.reset_index().merge(market_data.reset_index(),
                                      on=['trade_time', 'code'])

        evaluate1 = FactorEvaluate1(factor_data=dt1,
                                    factor_name=f,
                                    ret_name='nxt1_ret_{0}h'.format(period),
                                    roll_win=15,
                                    fee=0.0,
                                    scale_method='raw',
                                    expression=f,
                                    resampling_win=period,
                                    name=f)
        evaluate1.run()
        evaluate1.plot_results()
        evaluate1.save_results(base_output_dir=os.path.join(
            'temp3', task_id, str(period), class_name, '1'))


if __name__ == '__main__':
    run(method='ricso2', instruments='rbb', period=5, task_id='113001')
