import os, copy, pdb
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

from chaosmind.timing.sirius0003.workflow import WorkFlow
from lib.cux001 import FactorEvaluate1
from lib.rl012.predict import SignalGenerator
from kdutils.tactix import Tactix
from lib.uvx import *
from kdutils.macro2 import *


def _sanitize_frame(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    bad_mask = ~np.isfinite(df[cols].to_numpy(dtype=np.float64))
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        print(f"[WARN] 数据中发现 {bad_count} 个 NaN/Inf，已填充为 0.0")
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def load_data0(method, instruments, task_id, period, features, regime,
               ret_name, category):
    base_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
                             str(task_id), str(period), 'rl', 'data')
    if category == 'train':
        data = pd.read_feather(os.path.join(base_dirs, "train_data.feather"))
    elif category == 'val':
        data = pd.read_feather(os.path.join(base_dirs, "val_data.feather"))
    elif category == 'test':
        data = pd.read_feather(os.path.join(base_dirs, "test_data.feather"))

    data.rename(columns={ret_name: "nxt1_ret"}, inplace=True)
    data = data[['trade_time', 'code', 'nxt1_ret'] + features + regime]
    data = data.sort_values('trade_time').reset_index(drop=True)
    data = _sanitize_frame(data, ['nxt1_ret'] + features + regime)
    if data['code'].nunique() != 1:
        raise ValueError(
            f"test_data 不是单标的，检测到 {data['code'].nunique()} 个 code")
    return data


def predict(method, instruments, task_id, period, model_id):
    file_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
                             str(task_id), str(period), 'rl', 'pro',
                             str(model_id))
    output_dir = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'composite',
                              "model", "rl", str(model_id), "data", "or")
    best_model_path = os.path.join(file_dirs, "best_model")
    config_path = os.path.join(file_dirs, "config.json")
    generator = SignalGenerator(model_path=best_model_path,
                                config_path=config_path,
                                deterministic=True)
    for category in ['val', 'test']:
        ### 数据本身已经做过标准化，方向调整
        filename = os.path.join(output_dir,
                                "{0}_data.feather".format(category))
        data = load_data0(
            method=method,
            instruments=instruments,
            period=period,
            task_id=task_id,
            ret_name="nxt1_ret_{0}h".format(1),  ## 强化学习环境做了收益累计
            features=generator.config['features'],
            regime=[],
            category=category)
        filename = os.path.join(output_dir,
                                "{0}_data.feather".format(category))
        signals_df = generator.predict_signals(data)
        os.makedirs(output_dir, exist_ok=True)
        print(filename)
        signals_df.to_feather(filename)


def forecast(method, instruments, task_id, period, model_id):
    output_dir = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl', 'composite',
                              "model", "rl", str(model_id), "data", "wf")

    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=str(model_id))

    pdb.set_trace()
    workflow = WorkFlow(
        directory=params['model_path'],
        code=INSTRUMENTS_CODES[instruments],
        symbol="{0}9999".format(instruments.lower()),
        task_id=task_id,
        factors_infos=factors_infos,
        softmax_temperature=params['softmax_temperature'],
        min_open_signal_abs=params['min_open_signal_abs'],
        period=params['horizon'],  # 当前未使用上
        signal_method=params['signal_method'],  # 当前未使用上
        signal_params=params['signal_params'],  # 当前未使用上
        method=params['method'],  # 当前未使用上
        win=params['win']  # 当前未使用上
    )
    for category in ['val', 'test']:
        ### 数据本身已经做过标准化，方向调整
        data = load_data0(
            method=method,
            instruments=instruments,
            period=period,
            task_id=task_id,
            ret_name="nxt1_ret_{0}h".format(1),  ## 强化学习环境做了收益累计
            features=workflow.features,
            regime=[],
            category=category)
        # data = data.loc[:500]
        total_data1 = data.set_index(['trade_time', 'code'])
        all_trade_times = total_data1.index.get_level_values(
            'trade_time').unique().sort_values()
        res = []
        for time in all_trade_times:
            print(time)
            rt = workflow.create_values(trade_time=time, data=total_data1)
            res.append(rt)
        signals_df = pd.DataFrame(res)
        filename = os.path.join(output_dir,
                                "{0}_data.feather".format(category))
        os.makedirs(output_dir, exist_ok=True)
        print(filename)
        signals_df.to_feather(filename)


def metrics(method, instruments, task_id, period, model_id):

    def _metrics(dirs1, instruments, name, metrics_path, category):
        pdb.set_trace()
        base_dirs1 = os.path.join(dirs1, 'data')

        base_dirs2 = os.path.join(dirs1, 'composite', "model", "rl",
                                  str(model_id), "data", name)
        predict_data = pd.read_feather(
            os.path.join(base_dirs2, "{0}_data.feather".format(category)))
        data1 = pd.read_feather(
            os.path.join(base_dirs1, "{0}_data.feather".format(category)))

        predict_data['code'] = INSTRUMENTS_CODES[instruments]
        predict_data1 = predict_data.merge(data1,
                                           on=['trade_time', 'code'])
        evaluate1 = FactorEvaluate1(factor_data=predict_data1.reset_index(),
                                    factor_name='net_er_out',
                                    ret_name='nxt1_ret_{0}h'.format(period),
                                    roll_win=15,
                                    fee=0.000,
                                    scale_method='raw',
                                    expression="{0}_{1}".format(name,category),
                                    resampling_win=period,
                                    name=category)
        _ = evaluate1.run()
        evaluate1.plot_results()
        evaluate1.save_results(base_output_dir=os.path.join(metrics_path, name))
        
    dirs1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                         str(task_id), str(period), 'rl')
    metrics_path = os.path.join(dirs1, 'composite', "model", "rl",
                                    str(model_id), "metrics")
    for name in ['or', 'wf']:
        for category in ['val', 'test']:
            _metrics(dirs1=dirs1, 
                     instruments=instruments, 
                     name=name, metrics_path=metrics_path, 
                     category=category)
            

# def metrics(method, instruments, task_id, period, model_id):

#     def _metrics(predict_data, test_data1, instruments, name, metrics_path):
#         predict_data['code'] = INSTRUMENTS_CODES[instruments]
#         predict_data1 = predict_data.merge(test_data1,
#                                            on=['trade_time', 'code'])
#         evaluate1 = FactorEvaluate1(factor_data=predict_data1.reset_index(),
#                                     factor_name='net_er_out',
#                                     ret_name='nxt1_ret_{0}h'.format(period),
#                                     roll_win=15,
#                                     fee=0.000,
#                                     scale_method='raw',
#                                     expression=name,
#                                     resampling_win=period,
#                                     name=name)
#         _ = evaluate1.run()
#         evaluate1.plot_results()
#         evaluate1.save_results(base_output_dir=metrics_path)

#     dirs1 = os.path.join(base_path, method, instruments, 'temp', 'model',
#                          str(task_id), str(period), 'rl')
#     base_dirs1 = os.path.join(dirs1, 'data')

#     base_dirs2 = os.path.join(dirs1, 'composite', "model", "rl", str(model_id),
#                               "data")

#     for category in ['val', 'test']:
#         or_predict_data = pd.read_feather(
#             os.path.join(base_dirs2, "or_{0}_data.feather".format(category)))
#         wf_predict_data = pd.read_feather(
#             os.path.join(base_dirs2, "wf_{0}_data.feather".format(category)))

#         data1 = pd.read_feather(
#             os.path.join(base_dirs1, "{0}_data.feather".format(category)))

#         metrics_path = os.path.join(dirs1, 'composite', "model", "rl",
#                                     str(model_id), "metrics")

#         _metrics(predict_data=or_predict_data,
#                  test_data1=data1,
#                  instruments=instruments,
#                  name='or_{0}'.format(category),
#                  metrics_path=metrics_path)

#         _metrics(predict_data=wf_predict_data,
#                  test_data1=data1,
#                  instruments=instruments,
#                  name='wf_{0}'.format(category),
#                  metrics_path=metrics_path)

    # or_predict_data['code'] = INSTRUMENTS_CODES[instruments]
    # wf_predict_data['code'] = INSTRUMENTS_CODES[instruments]

    # base_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
    #                          str(task_id), str(period), 'rl', 'data')
    # test_data = pd.read_feather(os.path.join(base_dirs, "test_data.feather"))

    # test_data1 = test_data[[
    #     'trade_time', 'code', 'nxt1_ret_{0}h'.format(period)
    # ]]
    # pdb.set_trace()
    # or_predict_data = or_predict_data.merge(test_data1,
    #                                         on=['trade_time', 'code'])
    # evaluate1 = FactorEvaluate1(factor_data=or_predict_data.reset_index(),
    #                             factor_name='net_er_out',
    #                             ret_name='nxt1_ret_{0}h'.format(period),
    #                             roll_win=15,
    #                             fee=0.000,
    #                             scale_method='raw',
    #                             expression='or',
    #                             resampling_win=period,
    #                             name='or')
    # _ = evaluate1.run()
    # ## 读取收益率

    # wf_predict_data = wf_predict_data.merge(test_data1,
    #                                         on=['trade_time', 'code'])
    # evaluate2 = FactorEvaluate1(factor_data=wf_predict_data.reset_index(),
    #                             factor_name='net_er_out',
    #                             ret_name='nxt1_ret_{0}h'.format(period),
    #                             roll_win=15,
    #                             fee=0.000,
    #                             scale_method='raw',
    #                             expression='or',
    #                             resampling_win=period,
    #                             name='or')
    # _ = evaluate2.run()


if __name__ == '__main__':
    pdb.set_trace()
    variant = Tactix().start()

    if variant.form == "predict":
        predict(method=variant.method,
                instruments=variant.instruments,
                task_id=variant.task_id,
                period=variant.period,
                model_id=variant.model_id)

    elif variant.form == "forecast":
        forecast(method=variant.method,
                 instruments=variant.instruments,
                 task_id=variant.task_id,
                 period=variant.period,
                 model_id=variant.model_id)

    elif variant.form == 'metrics':
        metrics(method=variant.method,
                instruments=variant.instruments,
                task_id=variant.task_id,
                period=variant.period,
                model_id=variant.model_id)
