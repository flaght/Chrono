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

DEFAULT_SIGNAL_CANDIDATES = {
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


def _parse_config(value, name):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}不是合法JSON: {exc}") from exc
    return value


def _normalize_asset_specs(asset_specs, instruments, task_id):
    asset_specs = _parse_config(asset_specs, "asset_specs")
    if not isinstance(asset_specs, (list, tuple)) or len(asset_specs) < 2:
        raise ValueError("asset_specs必须包含至少两个品种")
    result = []
    for index, asset in enumerate(asset_specs):
        if not isinstance(asset, dict):
            raise TypeError(f"asset_specs[{index}]必须是字典")
        asset_instruments = str(asset.get("instruments", "")).strip()
        asset_task_id = str(asset.get("task_id", "")).strip()
        code = asset.get("code")
        if code is None:
            code = INSTRUMENTS_CODES.get(asset_instruments)
        code = str(code or "").strip().upper()
        if not asset_instruments or not asset_task_id or not code:
            raise ValueError(f"asset_specs[{index}]缺少code、instruments或task_id")
        result.append({"code": code, "instruments": asset_instruments,
                       "task_id": asset_task_id})
    if result[0]["instruments"] != str(instruments) or result[0]["task_id"] != str(task_id):
        raise ValueError("asset_specs第一项必须对应主调用的instruments和task_id")
    for key in ("code", "instruments", "task_id"):
        values = [asset[key] for asset in result]
        if len(values) != len(set(values)):
            raise ValueError(f"asset_specs中的{key}不能重复: {values}")
    return result


def _normalize_seed_run_ids(seed_run_ids):
    seed_run_ids = _parse_config(seed_run_ids, "seed_run_ids")
    if not isinstance(seed_run_ids, dict) or len(seed_run_ids) < 2:
        raise ValueError("seed_run_ids必须是至少包含两个seed的字典")
    result = {int(seed): str(run_id).strip()
              for seed, run_id in seed_run_ids.items()}
    if len(result) != len(seed_run_ids) or any(not run_id for run_id in result.values()):
        raise ValueError("seed_run_ids存在重复seed或空run_id")
    return result


def _normalize_signal_functions(value, name):
    value = _parse_config(value, name)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name}必须是非空的信号方法参数字典")
    for method, candidates in value.items():
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError(f"{name}.{method}必须包含至少一组参数")
        for signal_id, params in candidates.items():
            if not isinstance(params, dict):
                raise TypeError(f"{name}.{method}.{signal_id}必须是参数字典")
    return value


def load_er_data2(instruments, task_id, trial_id, base_path, dataset,
                  asset_specs, seed_run_ids):

    def filter_data(
            data,
            code,
            name,
            columns=['trade_time', 'code', 'predicted_z', 'future_ret_h']):
        data = data[data['code'] == code][columns]
        data = data.reset_index(drop=True)
        if data.empty:
            raise ValueError(f"{name}预测文件中没有品种{code}的数据")
        data.name = name
        return data

    assets = _normalize_asset_specs(asset_specs, instruments, task_id)
    ids = _normalize_seed_run_ids(seed_run_ids)
    if dataset not in ("val", "test"):
        raise ValueError("dataset只能是val或test")
    ensemble_name = "_".join(ids[seed] for seed in sorted(ids))
    dirs_path = os.path.join(
        base_path, str(trial_id), "hybrid_transformer_loss", "result",
        "ensemble", ensemble_name, dataset)
    prediction_paths = [Path(dirs_path) / f"seed_{seed}_predictions.csv"
                        for seed in sorted(ids)]
    prediction_paths.append(Path(dirs_path) / "ensemble_equal_predictions.csv")
    missing = [str(path) for path in prediction_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少预测文件: " + ", ".join(missing))

    data_sets = []
    for prediction_path in prediction_paths:
        model_name = prediction_path.stem.replace("_predictions", "")
        if model_name == "ensemble_equal":
            model_name = "ensemble"
        prediction_data = pd.read_csv(prediction_path)
        required = {"trade_time", "code", "predicted_z", "future_ret_h"}
        if not required.issubset(prediction_data.columns):
            raise ValueError(f"{prediction_path}缺少列: {sorted(required - set(prediction_data.columns))}")
        for asset in assets:
            data_sets.append(filter_data(prediction_data, asset["code"], model_name))
    print("Loaded predictions:", dirs_path,
          "models=", sorted({data.name for data in data_sets}),
          "codes=", [asset["code"] for asset in assets])
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


def _save_holdout_signals(signal_data, output_dir, signal_method,
                          signal_id):
    """保存测试集回测实际使用的逐 bar 信号，供实盘回放逐行核对。"""
    columns = ["trade_time", "code", "transformed", "signal",
               "future_ret_h"]
    missing = set(columns) - set(signal_data.columns)
    if missing:
        raise ValueError(f"holdout 信号缺少字段: {sorted(missing)}")
    records = signal_data[columns].rename(
        columns={"transformed": "predicted_z"}).copy()
    records["signal_method"] = str(signal_method)
    records["signal_id"] = str(signal_id)
    if records[["trade_time", "code"]].isna().any().any() or (
            records.duplicated(["trade_time", "code"]).any()):
        raise ValueError("holdout 信号的 trade_time/code 缺失或重复")
    path = Path(output_dir) / "signals.csv"
    csv_text = records.to_csv(index=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != csv_text:
            raise FileExistsError(f"已有不同的 holdout 信号文件，不覆盖: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text, encoding="utf-8")
    print(f"Holdout signals saved to: {path} rows={len(records)}")
    return path


def metrics_signal(method, instruments, task_id, period, trial_id,
                   asset_specs, seed_run_ids, selected_signals, fee=0.0):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    ids = _normalize_seed_run_ids(seed_run_ids)
    ensemble_name = "_".join(ids[seed] for seed in sorted(ids))
    data_sets = load_er_data2(instruments, task_id, trial_id, base_path1,
                              'test', asset_specs, ids)
    signal_functions1 = _normalize_signal_functions(selected_signals,
                                                    "selected_signals")
    for key1, functions in signal_functions1.items():
        for key2, params in functions.items():
            for data in data_sets:
                output_dirs = os.path.join(base_path1, str(trial_id),
                                           "holdout", ensemble_name, data.name)
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

                if signal_data.empty:
                    raise ValueError(f"测试集信号为空: model={data.name} method={key1}")
                code = signal_data.iloc[0]['code']
                name1 = f"{code}_{key1}_{key2}_test"
                evaluate1 = FactorEvaluate1(factor_data=signal_data,
                                            code=code,
                                            factor_name="signal",
                                            ret_name="future_ret_h",
                                            roll_win=15,
                                            fee=float(fee),
                                            scale_method="raw",
                                            expression=name1,
                                            name=name1,
                                            resampling_win=5)
                evaluate1.run()
                evaluate1.plot_results()
                result_dir = os.path.join(output_dirs, key1, str(key2))
                evaluate1.save_results(result_dir)
                _save_holdout_signals(signal_data,
                                      os.path.join(result_dir, name1),
                                      key1, key2)


## 校验70%用于选参数， 30% 用于冻结参数
def create_signal(method, instruments, task_id, period, trial_id,
                  asset_specs, seed_run_ids, signal_candidates=None,
                  fee=0.0):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    ids = _normalize_seed_run_ids(seed_run_ids)
    ensemble_name = "_".join(ids[seed] for seed in sorted(ids))
    data_sets = load_er_data2(instruments, task_id, trial_id, base_path1,
                              'val', asset_specs, ids)
    # 所有模型和品种共用同一个时间切割点，避免按各自行数切割后区间错位。
    unique_times = pd.concat(
        [pd.to_datetime(data["trade_time"]) for data in data_sets],
        ignore_index=True).drop_duplicates().sort_values().reset_index(
            drop=True)
    if len(unique_times) < 2:
        raise ValueError("校验集不足两个时间点，无法划分70%/30%")
    split_time = unique_times.iloc[int(len(unique_times) * 0.7)]

    candidates = (DEFAULT_SIGNAL_CANDIDATES if signal_candidates is None
                  else _normalize_signal_functions(signal_candidates,
                                                   "signal_candidates"))
    for key1, functions in candidates.items():
        for key2, params in functions.items():
            for data in data_sets:
                output_dirs = os.path.join(base_path1, str(trial_id),
                                           "composite", ensemble_name,
                                           data.name)
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
                    if data1.empty:
                        raise ValueError(f"{name}信号为空: model={data.name} method={key1}")
                    code = data1.iloc[0]['code']
                    name1 = "{0}_{1}_{2}_{3}".format(code, key1, key2, name)
                    evaluate1 = FactorEvaluate1(factor_data=data1,
                                                code=code,
                                                factor_name="signal",
                                                ret_name="future_ret_h",
                                                roll_win=15,
                                                fee=float(fee),
                                                scale_method="raw",
                                                expression=name1,
                                                name=name1,
                                                resampling_win=5)
                    stats = evaluate1.run()
                    evaluate1.plot_results()
                    evaluate1.save_results(
                        os.path.join(output_dirs, key1, str(key2)))


def _cli_parser():
    parser = Tactix().params()
    parser.add_argument("--asset_specs", type=json.loads, default=None)
    parser.add_argument("--seed_run_ids", type=json.loads, default=None)
    parser.add_argument("--signal_candidates", type=json.loads, default=None)
    parser.add_argument("--selected_signals", type=json.loads, default=None)
    parser.add_argument("--signal_eval_fee", type=float, default=None)
    return parser


def _required_variant(variant, name):
    value = getattr(variant, name, None)
    if value is None:
        raise ValueError(f"缺少{name}；请在YAML或命令行中显式传入")
    return value


if __name__ == '__main__':
    variant = Tactix().start(callback=_cli_parser)
    if variant.form == 'build':
        create_signal(method=variant.method,
                      instruments=variant.instruments,
                      task_id=variant.task_id,
                      period=variant.period,
                      trial_id=variant.trial_id,
                      asset_specs=_required_variant(variant, "asset_specs"),
                      seed_run_ids=_required_variant(variant, "seed_run_ids"),
                      signal_candidates=getattr(variant, "signal_candidates", None),
                      fee=getattr(variant, "signal_eval_fee", None) or 0.0)

    elif variant.form == 'metrics':
        metrics_signal(method=variant.method,
                       instruments=variant.instruments,
                       task_id=variant.task_id,
                       period=variant.period,
                       trial_id=variant.trial_id,
                       asset_specs=_required_variant(variant, "asset_specs"),
                       seed_run_ids=_required_variant(variant, "seed_run_ids"),
                       selected_signals=_required_variant(variant, "selected_signals"),
                       fee=getattr(variant, "signal_eval_fee", None) or 0.0)
    else:
        raise ValueError(f"不支持form={variant.form}，只能使用build或metrics")
