import json
from dotenv import load_dotenv
import multiprocessing as mp

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.bck001.engine import create_signal as create_signal_method
from lib.bck001.common import *
from lib.cux004 import FactorEvaluate1

PAIRE_TASK = {"113001": ("hcb", "134001")}

signal_functions = {
    "threshold_signal": {
        "1001": {
            "roll_num": 0,
            "threshold": 0.05
        }
    }
}


def load_er_data2(instruments, task_id, base_path, dataset):

    def filter_data(
            data,
            instrument,
            name,
            columns=['trade_time', 'code', 'predicted_z', 'future_ret_h']):
        data = data[data['code'] == INSTRUMENTS_CODES[instrument]][columns]
        data = data.reset_index(drop=True)
        data.name = name
        return data

    base_path1 = base_path

    ## 临时加载已经预测好的值和对应收益率
    dirs_path = os.path.join(
        base_path1,
        "hybrid_transformer_loss/result/ensemble/s42_1048239198485335_s2026_1026780123442923_s3407_1056628310147615/{0}"
        .format(dataset))

    ensemble_equal_data = pd.read_csv(
        os.path.join(dirs_path, "ensemble_equal_predictions.csv"))
    seed_42_data = pd.read_csv(
        os.path.join(dirs_path, "seed_42_predictions.csv"))

    seed_2026_data = pd.read_csv(
        os.path.join(dirs_path, "seed_2026_predictions.csv"))

    seed_3407_data = pd.read_csv(
        os.path.join(dirs_path, "seed_3407_predictions.csv"))

    ### 拆分品种
    left_instrument = instruments
    right_instrument = PAIRE_TASK[task_id][0]
    pdb.set_trace()
    left_ensemble_equal_data = filter_data(ensemble_equal_data,
                                           left_instrument, "ensemble")
    right_ensemble_equal_data = filter_data(ensemble_equal_data,
                                            right_instrument, "ensemble")

    left_seed_42_data = filter_data(seed_42_data, left_instrument, "seed_42")
    right_seed_42_data = filter_data(seed_42_data, right_instrument, "seed_42")

    left_seed_2026_data = filter_data(seed_2026_data, left_instrument,
                                      "seed_2026")
    right_seed_2026_data = filter_data(seed_2026_data, right_instrument,
                                       "seed_2026")

    left_seed_3407_data = filter_data(seed_3407_data, left_instrument,
                                      "seed_3407")
    right_seed_3407_data = filter_data(seed_3407_data, right_instrument,
                                       "seed_3407")

    data_sets = [
        left_ensemble_equal_data, right_ensemble_equal_data, left_seed_42_data,
        right_seed_42_data, left_seed_2026_data, right_seed_2026_data,
        left_seed_3407_data, right_seed_3407_data
    ]
    return data_sets


## 校验70%用于选参数， 30% 用于冻结参数
def create_signal(method, instruments, task_id, period, composite_method,
                  composite_id):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    data_sets = load_er_data2(instruments=instruments,
                              task_id=task_id,
                              base_path=base_path1,
                              dataset='val')
    ## 先转信号 再切割
    pdb.set_trace()
    #output_dirs = os.path.join(base_path1, "composite")
    for key1, functions in signal_functions.items():
        for key2, params in functions.items():
            for data in data_sets:
                output_dirs = os.path.join(base_path1, "composite", data.name)
                os.makedirs(output_dirs, exist_ok=True)
                signal_input = data.copy()
                signal_input["trade_time"] = pd.to_datetime(
                    signal_input["trade_time"])

                signal_input = (signal_input.rename(
                    columns={
                        "predicted_z": "transformed"
                    }).drop_duplicates(["trade_time", "code"]))
                signal_dt = create_signal_method(data=signal_input.copy(),
                                                 signal_method=key1,
                                                 name='transformed',
                                                 signal_params=params)
                signal_data = signal_input.merge(signal_dt,
                                                 on=['trade_time', 'code'])

                split_idx = int(len(signal_data) * 0.7)
                optimi_signal_data = signal_data.iloc[:split_idx].copy()
                obser_signal_data = signal_data.iloc[split_idx:].copy()
                optimi_signal_data = optimi_signal_data.reset_index(drop=True)
                obser_signal_data = obser_signal_data.reset_index(drop=True)

                ### 切割信号
                for name, data1 in zip(
                    ['optimi', 'obse'],
                    [optimi_signal_data, obser_signal_data]):
                    code = data1.loc[0]['code']
                    name1 = "{0}_{1}_{2}_{3}".format(code, key1, key2, name)
                    evaluate1 = FactorEvaluate1(factor_data=data1,
                                                code=code,
                                                factor_name="signal",
                                                ret_name="future_ret_h",
                                                roll_win=15,
                                                fee=0.0,
                                                scale_method="raw",
                                                expression=name1,
                                                name=name1,
                                                resampling_win=5)
                    _ = evaluate1.run()
                    evaluate1.plot_results()
                    evaluate1.save_results(
                        os.path.join(output_dirs, key1, key2))


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'build':
        create_signal(method=variant.method,
                      instruments=variant.instruments,
                      task_id=variant.task_id,
                      period=variant.period,
                      composite_method=0,
                      composite_id=1)
