import os, json, pdb, copy, pdb
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.data import filter_invalid_periods
from kdutils.macro2 import *
from lib.uvx import *
from kdutils.tactix import Tactix
from lib.syn001.linear import composit_equal1, composit_equal2
from chaosmind.timing.sirius1001.workflow import WorkFlow


def load_data2(method, instruments, task_id, period):
    base_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
                             str(task_id), str(period), 'rl', 'data')

    train_data = pd.read_feather(os.path.join(base_dirs, "train_data.feather"))

    val_data = pd.read_feather(os.path.join(base_dirs, "val_data.feather"))

    test_data = pd.read_feather(os.path.join(base_dirs, "test_data.feather"))
    return train_data, val_data, test_data


def equal_weight1(train_data, val_data, test_data, corr_data, corr, period,
                  basic_path):
    pdb.set_trace()
    train_result, train_evaluate1 = composit_equal1(
        data=train_data,
        selected_features=corr_data['expression'].to_list(),
        roll_win=15,
        period=period,
        scale_method='raw',
        expression='train_{0}'.format(corr),
        name='train')

    val_result, val_evaluate1 = composit_equal1(
        data=val_data,
        selected_features=corr_data['expression'].to_list(),
        roll_win=15,
        period=period,
        scale_method='raw',
        expression='val_{0}'.format(corr),
        name='val')

    test_result, test_evaluate1 = composit_equal1(
        data=test_data,
        selected_features=corr_data['expression'].to_list(),
        roll_win=15,
        period=period,
        scale_method='raw',
        expression='test_{0}'.format(corr),
        name='test')

    dir_path = os.path.join(basic_path, str(int(corr)))
    metrics_path = os.path.join(dir_path, "metrics", 'or')
    data_path = os.path.join(dir_path, "data", 'or')
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    train_result.reset_index().to_feather(
        os.path.join(data_path, "train_data.feather"))
    val_result.reset_index().to_feather(
        os.path.join(data_path, "val_data.feather"))
    test_result.reset_index().to_feather(
        os.path.join(data_path, "test_data.feather"))

    train_evaluate1.plot_results()
    train_evaluate1.save_results(base_output_dir=metrics_path)

    val_evaluate1.plot_results()
    val_evaluate1.save_results(base_output_dir=metrics_path)

    test_evaluate1.plot_results()
    test_evaluate1.save_results(base_output_dir=metrics_path)


def equal_weight2(val_data, test_data, corr_data, corr, instruments, task_id,
                  period, basic_path):
    factors_infos = [{
        "formula": row.expression,
        "direction": 1 if row.ic_mean > 0 else -1
    } for row in corr_data.itertuples()]

    wf = WorkFlow(factors_infos=factors_infos,
                  code=INSTRUMENTS_CODES[instruments],
                  symbol="{0}9999".format(instruments.lower()),
                  task_id=task_id,
                  period=period,
                  signal_method=None,
                  signal_params=None,
                  method=None,
                  win=None)
    wf.initialization()

    val_result, val_evaluate1 = composit_equal2(
        wf=wf,
        data=val_data,
        roll_win=15,
        period=period,
        scale_method='raw',
        expression='val_{0}'.format(corr),
        name='val')

    test_result, test_evaluate1 = composit_equal2(
        wf=wf,
        data=test_data,
        roll_win=15,
        period=period,
        scale_method='raw',
        expression='test_{0}'.format(corr),
        name='test')

    dir_path = os.path.join(basic_path, str(int(corr)))
    metrics_path = os.path.join(dir_path, "metrics", "wf")
    data_path = os.path.join(dir_path, "data", "wf")
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    # train_result.reset_index().to_feather(
    #     os.path.join(data_path, "train_data.feather"))
    val_result.reset_index().to_feather(
        os.path.join(data_path, "val_data.feather"))
    test_result.reset_index().to_feather(
        os.path.join(data_path, "test_data.feather"))

    # train_evaluate1.plot_results()
    # train_evaluate1.save_results(base_output_dir=metrics_path)

    val_evaluate1.plot_results()
    val_evaluate1.save_results(base_output_dir=metrics_path)

    test_evaluate1.plot_results()
    test_evaluate1.save_results(base_output_dir=metrics_path)


def forecast_model(method, instruments, task_id, period, composite_method):
    train_data, val_data, test_data = load_data2(method=method,
                                                 instruments=instruments,
                                                 task_id=task_id,
                                                 period=period)
    basic_path = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dt_path = os.path.join(basic_path, 'blend', 'corr')

    file_path = Path(dt_path)
    for csv_file in file_path.rglob('*.csv'):
        corr_data = pd.read_csv(csv_file, index_col=0)  ## 相关性过滤的因子组
        name = csv_file.parts[-1].split('.')[0]
        train_data1 = train_data[
            ['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)] +
            corr_data['expression'].to_list()]
        val_data1 = val_data[
            ['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)] +
            corr_data['expression'].to_list()]
        test_data1 = test_data[
            ['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)] +
            corr_data['expression'].to_list()]

        if composite_method == "equal_weight":
            equal_weight2(val_data=val_data1,
                          test_data=test_data1,
                          corr_data=corr_data,
                          corr=name,
                          period=period,
                          instruments=instruments,
                          task_id=task_id,
                          basic_path=os.path.join(basic_path, "composite",
                                                  "linear", 'equal_weight'))
        # for category in ['val', 'test']:
        #     if category == 'val':
        #         total_data1 = val_data1.set_index(['trade_time', 'code'])
        #     elif category == 'test':
        #         total_data1 = test_data1.set_index(['trade_time', 'code'])
        #     all_trade_times = total_data1.index.get_level_values(
        #         'trade_time').unique().sort_values()
        #     res = []
        #     for time in all_trade_times[0:20]:
        #         print(time)
        #         rt = wf.create_values(trade_time=time, data=total_data1)
        #         res.append(rt)
        #     signals_df = pd.DataFrame(res)
        #     filename = os.path.join(output_dir,
        #                         "wf_{0}_data.feather".format(category))
        #     os.makedirs(output_dir, exist_ok=True)
        #     print(filename)
        #     signals_df.to_feather(filename)


def predict_model(method, instruments, task_id, period, composite_method):
    train_data, val_data, test_data = load_data2(method=method,
                                                 instruments=instruments,
                                                 task_id=task_id,
                                                 period=period)

    basic_path = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dt_path = os.path.join(basic_path, 'blend', 'corr')

    file_path = Path(dt_path)
    for csv_file in file_path.rglob('*.csv'):
        corr_data = pd.read_csv(csv_file, index_col=0)
        name = csv_file.parts[-1].split('.')[0]
        train_data1 = train_data[
            ['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)] +
            corr_data['expression'].to_list()]
        val_data1 = val_data[
            ['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)] +
            corr_data['expression'].to_list()]
        test_data1 = test_data[
            ['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)] +
            corr_data['expression'].to_list()]

        train_data1 = filter_invalid_periods(data=train_data1,
                                             instruments=instruments,
                                             time_name='trade_time')
        val_data1 = filter_invalid_periods(data=val_data1,
                                           instruments=instruments,
                                           time_name='trade_time')
        test_data1 = filter_invalid_periods(data=test_data1,
                                            instruments=instruments,
                                            time_name='trade_time')

        # train_data1 = filter_invalid_periods(
        #     data=train_data1,
        #     invalid_periods=FILTER_YEAR_MAPPING[
        #         INSTRUMENTS_CODES[instruments]])

        # val_data1 = filter_invalid_periods(data=val_data1,
        #                                    invalid_periods=FILTER_YEAR_MAPPING[
        #                                        INSTRUMENTS_CODES[instruments]])

        # test_data1 = filter_invalid_periods(
        #     data=test_data1,
        #     invalid_periods=FILTER_YEAR_MAPPING[
        #         INSTRUMENTS_CODES[instruments]])

        pdb.set_trace()
        if composite_method == 'equal_weight':
            equal_weight1(train_data=train_data1,
                          val_data=val_data1,
                          test_data=test_data1,
                          corr_data=corr_data,
                          corr=name,
                          period=period,
                          basic_path=os.path.join(basic_path, "composite",
                                                  "linear", 'equal_weight'))


if __name__ == '__main__':
    ### 等权 固定权重，波动率倒数加权 不需要训练模型，所以预测和评估放在一起。
    variant = Tactix().start()
    if variant.form == "predict":  ## 原始模式生成er
        predict_model(method=variant.method,
                      instruments=variant.instruments,
                      task_id=variant.task_id,
                      period=variant.period,
                      composite_method=variant.composite_method)
    elif variant.form == "forecast":  ## wf 模式生成er
        forecast_model(method=variant.method,
                       instruments=variant.instruments,
                       task_id=variant.task_id,
                       period=variant.period,
                       composite_method=variant.composite_method)
