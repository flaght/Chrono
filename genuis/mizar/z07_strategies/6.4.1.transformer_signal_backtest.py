import json
import os
from pathlib import Path
from dotenv import load_dotenv
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.bck001.engine import create_signal as create_signal_method
from lib.bck001.common import *
from lib.cux004 import FactorEvaluate1
from lib.flp001 import load_signal_performance

PAIRE_TASK = {"113001": ("hcb", "134001")}

ENSEMBLE_TASK = {
    10001: {
        42: "1048239198485335",
        3407: "1056628310147615",
        2026: "1026780123442923"
    },
    10002: {
        42: "1013995178499222",
        3407: "1077034256017865",
        2026: "1087298621243300"
    }
}

SIGNAL_FUNCTION_MAPPING = {
    10001: {
        "linear_threshold_signal": {
            "1204": {
                "roll_num": 0,
                "threshold": 0.05,
                "upper": 0.15,
            }
        },
        "threshold_signal": {
            "1304": {
                "roll_num": 0,
                "threshold": 0.075,
            }
        }
    },
    10002: {
        "linear_signal": {
            "1103": {
                "roll_num": 0,
                "threshold": 0.15,
            }
        },
        "threshold_signal": {
            "1301": {
                "roll_num": 0,
                "threshold": 0.02,
            }
        },
    }
}

signal_functions = {
    # 第一梯队A：连续线性仓位
    # position = clip(predicted_z / threshold, -1, 1)
    "linear_signal": {
        "1101": {
            "roll_num": 0,
            "threshold": 0.05,
        },
        "1102": {
            "roll_num": 0,
            "threshold": 0.10,
        },
        "1103": {
            "roll_num": 0,
            "threshold": 0.15,
        },
    },

    # 第一梯队B：带死区的连续仓位
    # abs(er) <= threshold 时空仓；
    # abs(er) 从 threshold 增长到 upper 时，仓位从0线性增加到1。
    "linear_threshold_signal": {
        "1201": {
            "roll_num": 0,
            "threshold": 0.01,
            "upper": 0.075,
        },
        "1202": {
            "roll_num": 0,
            "threshold": 0.02,
            "upper": 0.10,
        },
        "1203": {
            "roll_num": 0,
            "threshold": 0.03,
            "upper": 0.125,
        },
        "1204": {
            "roll_num": 0,
            "threshold": 0.05,
            "upper": 0.15,
        },
    },

    # 第一梯队C：固定阈值离散仓位，作为对照
    # 输出{-1, 0, 1}
    "threshold_signal": {
        "1301": {
            "roll_num": 0,
            "threshold": 0.02,
        },
        "1302": {
            "roll_num": 0,
            "threshold": 0.03,
        },
        "1303": {
            "roll_num": 0,
            "threshold": 0.05,
        },
        "1304": {
            "roll_num": 0,
            "threshold": 0.075,
        },
    },
}


def load_er_data2(instruments, task_id, trial_id, base_path, dataset):

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

    pdb.set_trace()
    dirs_path = os.path.join(
        base_path1, str(trial_id), "hybrid_transformer_loss", "result",
        "ensemble", "_".join(
            str(v) for _, v in sorted(ENSEMBLE_TASK[int(trial_id)].items())),
        dataset)
    ## 临时加载已经预测好的值和对应收益率
    # dirs_path = os.path.join(
    #     base_path1,
    #     "hybrid_transformer_loss/result/ensemble/s42_1048239198485335_s2026_1026780123442923_s3407_1056628310147615/{0}"
    #     .format(dataset))

    prediction_paths = sorted(Path(dirs_path).glob("*_predictions.csv"))
    if not prediction_paths:
        raise FileNotFoundError(f"没有找到预测文件: {dirs_path}")

    # 自动加载目录中的全部模型，不把模型数量写死为4或5。
    left_instrument = instruments
    right_instrument = PAIRE_TASK[task_id][0]
    data_sets = []
    for prediction_path in prediction_paths:
        model_name = prediction_path.stem.replace("_predictions", "")
        if model_name == "ensemble_equal":
            model_name = "ensemble"
        prediction_data = pd.read_csv(prediction_path)
        data_sets.append(
            filter_data(prediction_data, left_instrument, model_name))
        data_sets.append(
            filter_data(prediction_data, right_instrument, model_name))
    print("Loaded prediction models:",
          sorted({data.name
                  for data in data_sets}))
    return data_sets


def save_model_comparison(records, output_dir):
    """保存所有模型、品种和数据段的四项核心信号绩效。"""
    comparison = pd.DataFrame(records)
    if comparison.empty:
        raise ValueError("没有可汇总的信号评估结果")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "signal_model_comparison.csv")
    comparison.to_csv(csv_path, index=False)

    metrics = [
        ("sharpe2", "Annualized Sharpe"),
        ("calmar", "Calmar Ratio"),
        ("win_rate", "Active Win Rate"),
        ("profit_ratio", "Profit/Loss Ratio"),
    ]
    for segment, segment_data in comparison.groupby("segment", sort=False):
        fig, axes = plt.subplots(2, 2, figsize=(18, 11))
        fig.suptitle(f"Signal Model Comparison | segment={segment}",
                     fontsize=17)
        for ax, (metric, title) in zip(axes.flat, metrics):
            table = segment_data.pivot(
                index=["signal_method", "signal_id", "model"],
                columns="code",
                values=metric)
            table.plot.bar(ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Model")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.3)
            if metric == "win_rate":
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
                ax.set_ylim(0, 1)
            for container in ax.containers:
                labels = []
                for value in container.datavalues:
                    if not np.isfinite(value):
                        labels.append("")
                    elif metric == "win_rate":
                        labels.append(f"{value:.1%}")
                    else:
                        labels.append(f"{value:.2f}")
                ax.bar_label(container, labels=labels, padding=2, fontsize=8)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        image_path = os.path.join(output_dir,
                                  f"signal_model_comparison_{segment}.png")
        fig.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Signal comparison saved to: {csv_path}")
    return comparison


def query_signal_results(source, segment="obse"):
    """查询四指标矩阵；source可以是CSV路径、DataFrame或rl根目录。"""
    source_path = Path(source) if not isinstance(source,
                                                 pd.DataFrame) else None
    if isinstance(source, pd.DataFrame):
        data = source.copy()
    elif source_path.is_file():
        data = pd.read_csv(source_path)
    else:
        data = load_signal_performance(
            source_path, data_type=segment).rename(columns={
                "ann_sharpe": "sharpe2",
                "pl_ratio": "profit_ratio"
            })
    segment = "obse" if str(segment).lower() == "obs" else str(segment).lower()
    required = {
        "model", "code", "segment", "sharpe2", "calmar", "win_rate",
        "profit_ratio"
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("汇总文件缺少字段: " + ", ".join(sorted(missing)))
    selected = data[data["segment"].replace({"obs": "obse"}) == segment].copy()
    if selected.empty:
        raise ValueError(f"没有segment={segment}的评估结果")
    return selected.pivot(
        index=["signal_method", "signal_id", "model"],
        columns="code",
        values=["sharpe2", "calmar", "win_rate", "profit_ratio"],
    ).sort_index()


def metrics_signal(method, instruments, task_id, period, trial_id):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    data_sets = load_er_data2(instruments=instruments,
                              task_id=task_id,
                               trial_id=trial_id,
                              base_path=base_path1,
                              dataset='test')
    # 所有模型和品种共用同一个时间切割点，避免按各自行数切割后区间错位。
    unique_times = pd.concat(
        [pd.to_datetime(data["trade_time"]) for data in data_sets],
        ignore_index=True).drop_duplicates().sort_values().reset_index(
            drop=True)
    signal_functions1 = SIGNAL_FUNCTION_MAPPING[int(trial_id)]
    for key1, functions in signal_functions1.items():
        for key2, params in functions.items():
            for data in data_sets:
                output_dirs = os.path.join(base_path1, str(trial_id), "holdout", data.name)
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
                signal_data = signal_data.sort_values("trade_time")

                ### 切割信号
                for name, data1 in zip(['test'], [signal_data]):
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
                    stats = evaluate1.run()
                    evaluate1.plot_results()
                    evaluate1.save_results(
                        os.path.join(output_dirs, key1, key2))


## 校验70%用于选参数， 30% 用于冻结参数
def create_signal(method, instruments, task_id, period, trial_id):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    data_sets = load_er_data2(instruments=instruments,
                              task_id=task_id,
                              trial_id=trial_id,
                              base_path=base_path1,
                              dataset='val')
    # 所有模型和品种共用同一个时间切割点，避免按各自行数切割后区间错位。
    unique_times = pd.concat(
        [pd.to_datetime(data["trade_time"]) for data in data_sets],
        ignore_index=True).drop_duplicates().sort_values().reset_index(
            drop=True)
    split_time = unique_times.iloc[int(len(unique_times) * 0.7)]

    for key1, functions in signal_functions.items():
        for key2, params in functions.items():
            for data in data_sets:
                output_dirs = os.path.join(base_path1, str(trial_id),
                                           "composite", data.name)
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
                signal_data = signal_data.sort_values("trade_time")

                optimi_signal_data = signal_data[signal_data["trade_time"] <=
                                                 split_time].copy()
                obser_signal_data = signal_data[signal_data["trade_time"] >
                                                split_time].copy()
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
                    stats = evaluate1.run()
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
                      trial_id=variant.trial_id)

    elif variant.form == 'metrics':
        metrics_signal(method=variant.method,
                       instruments=variant.instruments,
                       task_id=variant.task_id,
                       period=variant.period,
                       trial_id=variant.trial_id)
