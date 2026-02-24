import os,json,pdb,copy
import pandas as pd
from joblib import Parallel, delayed
from dotenv import load_dotenv
load_dotenv()

from kdutils.macro2 import *
from lib.uvx import *

from lib.cux002 import StrategyEvaluate1
from lib.cux001 import FactorEvaluate1
from kdutils.tactix import Tactix
from lumina.genetic.signal.method import *
from lib.svx001 import create_position

signal_functions = {
    'rollrank_signal': {
        "1001": {
            'roll_num': 20,
            'threshold': 0.7
        },
        "1002": {
            'roll_num': 40,
            'threshold': 0.7
        }
    },
    'adaptive_signal': {
        "1001": {
            'roll_num': 25,
            'threshold': 0.9
        }
    }
}

## 采用两种评估方式，另外一种好像对不上，需要处理
def evaluate(factor_data, param_id, signal_method, signal_params, strategy_method,
            strategy_params, strategy_settings,
            period, category, output_dirs):
    pos_data, total_data2 = create_position(predict_data=factor_data,
                                            signal_method=signal_method,
                                            signal_params=signal_params,
                                            strategy_method=strategy_method,
                                            strategy_params=strategy_params)
    total_data2 = total_data2.stack().reset_index()
    pos_data = pos_data.stack()
    pos_data.name = 'signal'
    total_data = total_data2.merge(pos_data.reset_index(), on=['trade_time','code'])
    evaluate = FactorEvaluate1(factor_data=total_data,
                                factor_name='signal',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=15,
                                fee=strategy_settings['commission'] * 2,
                                scale_method='raw',
                                expression='test',
                                resampling_win=15)
    state2 = evaluate.run()
    
    evaluate.plot_results()
    out_dirs = os.path.join(output_dirs, str(param_id), category)
    os.makedirs(out_dirs, exist_ok=True)
    evaluate.save_results(out_dirs)
    return {'total_ret': state2['total_ret'], 'avg_ret':state2['avg_ret'],
            'max_dd':state2['max_dd'],
            'calmar':state2['calmar'],
            'sharpe2':state2['sharpe2'],
            'turnover':state2['turnover'],
            'fee':strategy_settings['commission'],
            'param_id':param_id,
            'category':category}
    

def parallel_evaluate(train_data, test_data, strategy_settings, signal_info,period,output_dirs):
    train_state = evaluate(factor_data= train_data, 
                           param_id=signal_info['param_id'],
                           signal_method=signal_info['method'], 
                           signal_params=signal_info['param'], 
                           strategy_settings=strategy_settings,
                           strategy_method=None,
                           strategy_params=None, 
                           period=period, 
                           category='train', 
                           output_dirs=output_dirs)

    test_state = evaluate(factor_data= test_data, 
                           param_id=signal_info['param_id'],
                           signal_method=signal_info['method'], 
                           signal_params=signal_info['param'], 
                           strategy_settings=strategy_settings,
                           strategy_method=None,
                           strategy_params=None, 
                           period=period, 
                           category='test', 
                           output_dirs=output_dirs)
    total_data = pd.concat([train_data,test_data],axis=0).sort_values(by=['trade_time','code'])
    all_state = evaluate(factor_data= total_data, 
                           param_id=signal_info['param_id'],
                           signal_method=signal_info['method'], 
                           signal_params=signal_info['param'], 
                           strategy_settings=strategy_settings,
                           strategy_method=None,
                           strategy_params=None, 
                           period=period, 
                           category='all', 
                           output_dirs=output_dirs)
    return pd.DataFrame([train_state, test_state, all_state])
    
    

def train_signal(method, instruments, task_id, period, name, 
                 syns_id,
                 model_name):
    strategy_settings = {
        'commission':  COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        'slippage': 0,
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }
    pdb.set_trace()
    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research")
    factors_sub = os.path.join("result", model_name, str(syns_id), "factors")
    train_factors = pd.read_feather(os.path.join(outdirs, factors_sub, "train.feather"))
    test_factors = pd.read_feather(os.path.join(outdirs, factors_sub, "test.feather"))
    
    file_dirs = os.path.join(base_path, method, instruments,
                            "temp","model",str(task_id), 
                            str(period), "signals")
    output_dirs = os.path.join(outdirs, "signal", "results", name, model_name)
    os.makedirs(output_dirs, exist_ok=True)
    params_list = load_params3(file_dirs=file_dirs, file_name='signal', signal_name='quantile')
    popilations = Parallel(n_jobs=4, verbose=1)(
        delayed(parallel_evaluate)(
            train_data=train_factors, 
            test_data=test_factors, 
            strategy_settings=strategy_settings, 
            signal_info=params_list[i],
            period=period,
            output_dirs=output_dirs
        ) for i in range(0, len(params_list))
    )
    results = pd.concat(popilations).reset_index(drop=True)
    results.to_csv(os.path.join(output_dirs, "results.csv"))
    
    
    

if __name__ == '__main__':
    variant = Tactix().start()
    train_signal(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name, 
                    syns_id=variant.syns_id,
                    model_name=variant.model_name)