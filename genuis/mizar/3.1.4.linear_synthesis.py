import os, json, pdb, copy
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.syn001.linear import composit_equal


def load_data2(method, instruments, task_id, period):
    base_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
                             str(task_id), str(period), 'rl', 'data')

    train_data = pd.read_feather(os.path.join(base_dirs, "train_data.feather"))

    val_data = pd.read_feather(os.path.join(base_dirs, "val_data.feather"))

    test_data = pd.read_feather(os.path.join(base_dirs, "test_data.feather"))
    return train_data, val_data, test_data


def equal_weight(train_data, val_data, test_data, corr_data, corr, period,
                 basic_path):

    train_result, train_evaluate1 = composit_equal(
        data=train_data,
        selected_features=corr_data['expression'].to_list(),
        roll_win=15,
        period=period,
        scale_method='raw',
        expression='train_{0}'.format(corr),
        name='train')

    val_result, val_evaluate1 = composit_equal(
        data=val_data,
        selected_features=corr_data['expression'].to_list(),
        roll_win=15,
        period=period,
        scale_method='raw',
        expression='val_{0}'.format(corr),
        name='val')

    test_result, test_evaluate1 = composit_equal(
        data=test_data,
        selected_features=corr_data['expression'].to_list(),
        roll_win=15,
        period=period,
        scale_method='raw',
        expression='test_{0}'.format(corr),
        name='test')

    dir_path = os.path.join(basic_path, str(int(corr)))
    metrics_path = os.path.join(dir_path, "metrics")
    data_path = os.path.join(dir_path, "data")
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


def train_model(method, instruments, task_id, period, form):
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
        if form == 'equal_weight':
            equal_weight(train_data=train_data1,
                         val_data=val_data1,
                         test_data=test_data1,
                         corr_data=corr_data,
                         corr=name,
                         period=period,
                         basic_path=os.path.join(basic_path, "composite",
                                                 "linear", 'equal_weight'))


if __name__ == '__main__':
    variant = Tactix().start()
    train_model(method=variant.method,
                instruments=variant.instruments,
                task_id=variant.task_id,
                period=variant.period,
                form=variant.form)
