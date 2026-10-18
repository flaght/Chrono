"""HybridTransformer-MSE 调用端：RB为主、HC为辅的多资产监督基线。"""

import copy
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.uvx import *
from lib.nn003.predict import predict_test_set
from lib.nn003.train import train_model


PAIRE_TASK = {"113001": ("hcb", "134001")}


def _sanitize_frame(df, columns):
    columns = [column for column in columns if column in df.columns]
    if not columns:
        return df
    df = df.copy()
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(df[columns].to_numpy(dtype=np.float64))
    if int(bad.sum()):
        print(f"[WARN] 数据中发现 {int(bad.sum())} 个 NaN/Inf，已填充为 0.0")
    df[columns] = df[columns].replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _load_split(method, instruments, task_id, period, split, features,
                regime, ret_name, expected_code):
    data_dir = os.path.join(base_path, method, instruments, "temp", "model",
                            str(task_id), str(period), "rl", "data")
    path = os.path.join(data_dir, f"{split}_data.feather")
    data = pd.read_feather(path)
    required = {"trade_time", "code", ret_name, *features, *regime}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    data = data.rename(columns={ret_name: "nxt1_ret"})
    data = data[["trade_time", "code", "nxt1_ret"] + features + regime]
    data["trade_time"] = pd.to_datetime(data["trade_time"], errors="raise")
    data = _sanitize_frame(
        data.sort_values("trade_time").reset_index(drop=True),
        features + regime)
    codes = set(data["code"].astype(str))
    if data.empty or codes != {expected_code}:
        raise ValueError(f"{path} 应只包含 {expected_code}，实际为 {sorted(codes)}")
    return data


def load_data1(method, instruments, task_id, period, features, regime,
               ret_name, expected_code):
    train = _load_split(method, instruments, task_id, period, "train",
                        features, regime, ret_name, expected_code)
    val = _load_split(method, instruments, task_id, period, "val",
                      features, regime, ret_name, expected_code)
    if train["trade_time"].max() >= val["trade_time"].min():
        raise ValueError(f"{expected_code} train/val 时间重叠或顺序错误")
    return train, val


def load_test_data(method, instruments, task_id, period, features, regime,
                   ret_name, expected_code):
    return _load_split(method, instruments, task_id, period, "test", features,
                       regime, ret_name, expected_code)


def _merge_assets(left, right):
    data = pd.concat([left, right], ignore_index=True)
    if data.duplicated(["code", "trade_time"]).any():
        raise ValueError("联合数据存在重复的 code/trade_time")
    return data.sort_values(["code", "trade_time"]).reset_index(drop=True)


def _load_params(method, instruments, task_id, period, env_id, trade_id,
                 model_id, train_id, feature_id, regime_id):
    file_dirs = os.path.join(base_path, method, instruments, "temp", "model",
                             str(task_id), str(period), "rl")
    return load_rl_params(
        file_dirs=file_dirs, trade_id=trade_id, model_id=model_id,
        feature_id=feature_id, env_id=env_id, train_id=train_id,
        regime_id=regime_id, name='htf')


def _run_identity(trade_params, env_params, model_params, train_params,
                  selected_features, min_regime, daily_regime):
    total = {"algorithm": "hybrid_transformer_mse_v1"}
    for values in (trade_params, env_params, model_params, train_params):
        total.update(copy.deepcopy(values))
    total.update(selected_features=selected_features,
                 min_regime=min_regime, daily_regime=daily_regime)
    return Params.create_tag(total)


def _validate_core(period, trade_params, env_params):
    if int(period) != 5 or trade_params["ret_name"] != "nxt1_ret_5h":
        raise ValueError(
            "HybridTransformer-MSE 要求 period=5、ret_name=nxt1_ret_5h")
    if not bool(env_params.get("use_hybrid_transformer", True)):
        raise ValueError("HybridTransformer-MSE 要求 use_hybrid_transformer=true")


def _result_dir(method, instruments, task_id, period, name):
    return os.path.join(base_path, method, instruments, "temp", "model",
                        str(task_id), str(period), "rl",
                        "hybrid_transformer_mse", "result", str(name))


def train(method, instruments, task_id, period, env_id, trade_id, model_id,
          train_id, feature_id, regime_id):
    task_key = str(task_id)
    if task_key not in PAIRE_TASK:
        raise KeyError(f"未配置主任务 {task_key} 的配对品种")
    (env_params, trade_params, model_params, train_params, selected_features,
     min_regime, daily_regime) = _load_params(
         method, instruments, task_id, period, env_id, trade_id, model_id,
         train_id, feature_id, regime_id)
    _validate_core(period, trade_params, env_params)
    name = _run_identity(
        trade_params, env_params, model_params, train_params,
        selected_features, min_regime, daily_regime)
    output_dir = _result_dir(method, instruments, task_id, period, name)
    os.makedirs(output_dir, exist_ok=True)
    logger.configure(log_file=os.path.join(output_dir, "model.log"))

    right_instruments, right_task = PAIRE_TASK[task_key]
    rb_train, rb_val = load_data1(
        method, instruments, task_id, period, selected_features, min_regime,
        trade_params["ret_name"], "RB")
    hc_train, hc_val = load_data1(
        method, right_instruments, right_task, period, selected_features,
        min_regime, trade_params["ret_name"], "HC")
    train_data = _merge_assets(rb_train, hc_train)
    val_data = _merge_assets(rb_val, hc_val)
    env_config = {
        "holding_period": int(env_params["holding_period"]),
        "prediction_bound_std": float(
            env_params.get("prediction_bound_std", 3.0)),
        "use_hybrid_transformer": True,
        "default_lookback": int(env_params.get("default_lookback", 20)),
        "asset_sampling": "balanced",
        "seed": int(env_params.get("seed", 42)),
    }
    logger.info(
        f"HybridTransformer-MSE训练集: {len(train_data)} 行，"
        f"RB={len(rb_train)} HC={len(hc_train)}")
    logger.info(
        f"HybridTransformer-MSE校验集: {len(val_data)} 行，"
        f"RB={len(rb_val)} HC={len(hc_val)}")
    logger.info(f"env_config: {env_config}")
    logger.info(f"model_params: {model_params}")
    logger.info(f"train_params: {train_params}")
    logger.info(f"selected_features: {selected_features}")
    return train_model(
        train_data, val_data, selected_features + min_regime, env_config,
        copy.deepcopy(model_params), copy.deepcopy(train_params), output_dir)


def predict(method, instruments, task_id, period, env_id, trade_id, model_id,
            train_id, feature_id, regime_id):
    task_key = str(task_id)
    if task_key not in PAIRE_TASK:
        raise KeyError(f"未配置主任务 {task_key} 的配对品种")
    (env_params, trade_params, model_params, train_params, selected_features,
     min_regime, daily_regime) = _load_params(
         method, instruments, task_id, period, env_id, trade_id, model_id,
         train_id, feature_id, regime_id)
    _validate_core(period, trade_params, env_params)
    name = _run_identity(
        trade_params, env_params, model_params, train_params,
        selected_features, min_regime, daily_regime)
    output_dir = _result_dir(method, instruments, task_id, period, name)
    right_instruments, right_task = PAIRE_TASK[task_key]
    rb = load_test_data(method, instruments, task_id, period,
                        selected_features, min_regime,
                        trade_params["ret_name"], "RB")
    hc = load_test_data(method, right_instruments, right_task, period,
                        selected_features, min_regime,
                        trade_params["ret_name"], "HC")
    model_path = os.path.join(output_dir, "models", "best_model.pt")
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"HybridTransformer-MSE 最佳模型不存在: {model_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"HybridTransformer-MSE 训练配置不存在: {config_path}")
    output_path = os.path.join(output_dir, "metrics", "test_results.csv")
    print(f"[HYBRID_TRANSFORMER_MSE_TEST_START] RB={len(rb)} HC={len(hc)}")
    return predict_test_set(
        model_path, config_path, _merge_assets(rb, hc), output_path,
        inference_batch_size=int(
            train_params.get("inference_batch_size", 4096)),
        device=str(model_params.get("device", "auto")))


if __name__ == "__main__":
    variant = Tactix().start()
    if variant.form == "train":
        train(method=variant.method, instruments=variant.instruments,
              task_id=variant.task_id, period=variant.period,
              env_id=variant.env_id, trade_id=variant.trade_id,
              model_id=variant.model_id, train_id=variant.train_id,
              feature_id=variant.feature_id, regime_id=variant.regime_id)
    elif variant.form == "predict":
        predict(method=variant.method, instruments=variant.instruments,
                task_id=variant.task_id, period=variant.period,
                env_id=variant.env_id, trade_id=variant.trade_id,
                model_id=variant.model_id, train_id=variant.train_id,
                feature_id=variant.feature_id, regime_id=variant.regime_id)
    else:
        raise ValueError(f"不支持 form={variant.form}")
