import os, pdb
import pandas as pd
from sklearn.linear_model import LinearRegression

from lib.syn001.base import train_model as base_train_model
from lib.lsx001 import fetch_times
from lib.cux001 import FactorEvaluate1
from kdutils.macro2 import *
from lib import logger


def equal(data, selected_features):
    data1 = data
    data1 = data1[selected_features].mean(axis=1)
    data1.name = 'transformed'
    return data1


## 直接合成
def composit_equal1(data,
                    selected_features,
                    roll_win,
                    period,
                    scale_method,
                    expression,
                    name=None):
    data = data.set_index(['trade_time', 'code'])
    data1 = equal(data=data, selected_features=selected_features)
    data1 = pd.concat([data1, data["nxt1_ret_{0}h".format(period)]],
                      axis=1).sort_values(by=['trade_time', 'code'])
    evaluate1 = FactorEvaluate1(
        factor_data=data1.reset_index(),
        factor_name='transformed',
        ret_name='nxt1_ret_{0}h'.format(period),
        roll_win=roll_win,
        fee=0.000,
        scale_method=scale_method,
        expression=expression,
        resampling_win=period,
        name=name if isinstance(name, str) else expression)
    _ = evaluate1.run()
    pdb.set_trace()
    return data1, evaluate1


## workflow
def composit_equal2(wf,
                    data,
                    roll_win,
                    period,
                    scale_method,
                    expression,
                    name=None):
    res = []
    data = data.set_index(['trade_time', 'code'])
    all_trade_times = data.index.get_level_values(
        'trade_time').unique().sort_values()
    for time in all_trade_times:
        print(time)
        rt = wf.create_values(trade_time=time, data=data)
        res.append(rt)
    data1 = pd.DataFrame(res)
    #pdb.set_trace()
    data1 = data1.set_index(['trade_time', 'code'])[['task_id', 'transformed']]
    data1 = pd.concat([data1, data["nxt1_ret_{0}h".format(period)]],
                      axis=1).sort_values(by=['trade_time', 'code'])
    evaluate1 = FactorEvaluate1(
        factor_data=data1.reset_index(),
        factor_name='transformed',
        ret_name='nxt1_ret_{0}h'.format(period),
        roll_win=roll_win,
        fee=0.000,
        scale_method=scale_method,
        expression=expression,
        resampling_win=period,
        name=name if isinstance(name, str) else expression)
    _ = evaluate1.run()
    return data1, evaluate1


def train_model(method, task_id, instruments, period, name):
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    filename = os.path.join(dirs, "final_{0}_data.feather".format(name))
    final_data = pd.read_feather(filename).set_index(['trade_time', 'code'])

    print(final_data.columns)
    final_data1 = final_data.drop(['nxt1_ret_{0}h'.format(period)], axis=1)
    final_data1 = final_data1.mean(axis=1)
    final_data1.name = 'predict'
    final_data1 = pd.concat(
        [final_data1, final_data[['nxt1_ret_{0}h'.format(period)]]], axis=1)
    test_data = final_data1.loc[
        time_array['test_time'][0]:time_array['test_time'][1]]

    test_data.reset_index().to_feather(
        os.path.join(dirs, "linear_{0}_data.feather".format(name)))


def train_model1(train_data, test_data, selected_features, params, roll_win,
                 period, outdirs):
    base_train_model(model_class=LinearRegression,
                     train_data=train_data,
                     test_data=test_data,
                     selected_features=selected_features,
                     params={},
                     roll_win=roll_win,
                     period=period,
                     outdirs=outdirs)


def train_model2(train_data, test_data, selected_features, roll_win, period,
                 outdirs):

    def calc(data, selected_features, period, category, outdirs):
        data = data.set_index(['trade_time', 'code'])
        data1 = equal(data=data, selected_features=selected_features)
        data1 = pd.concat([data1, data["nxt1_ret_{0}h".format(period)]],
                          axis=1)

        evaluate1 = FactorEvaluate1(factor_data=data1.reset_index(),
                                    factor_name='transformed',
                                    ret_name='nxt1_ret_{0}h'.format(period),
                                    roll_win=roll_win,
                                    fee=0.000,
                                    scale_method='raw',
                                    expression=category,
                                    resampling_win=15)
        state1 = evaluate1.run()
        logger.table(pd.DataFrame([state1]), title="{0} 绩效".format(category))
        evaluate1.plot_results()
        ## 保存因子值 用于转信号
        factors_path = os.path.join(outdirs, "factors")
        os.makedirs(factors_path, exist_ok=True)
        data1.reset_index().to_feather(
            os.path.join(factors_path, f"{category}.feather"))
        evaluate1.save_results(os.path.join(outdirs, category))

    calc(data=train_data,
         selected_features=selected_features,
         period=period,
         category='train',
         outdirs=outdirs)
    calc(data=test_data,
         selected_features=selected_features,
         period=period,
         category='test',
         outdirs=outdirs)
