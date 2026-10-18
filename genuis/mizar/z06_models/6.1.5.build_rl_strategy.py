"""rl016 调用端：RB 为主、HC 为辅的联合收益预测模型。"""

import copy
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.uvx import *
from lib.rl016.predict import predict_test_set
from lib.rl016.train import train_model


PAIRE_TASK = {"113001": ("hcb", "134001")}


def _sanitize_frame(df, columns):
    columns = [c for c in columns if c in df.columns]
    if not columns:
        return df
    df = df.copy()
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(df[columns].to_numpy(dtype=np.float64))
    if int(bad.sum()):
        print(f"[WARN] 数据中发现 {int(bad.sum())} 个 NaN/Inf，已填充为 0.0")
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
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
    data = data.sort_values("trade_time").reset_index(drop=True)
    data = _sanitize_frame(data, features + regime)
    codes = set(data["code"].astype(str))
    if data.empty or codes != {expected_code}:
        raise ValueError(f"{path} 应只包含 {expected_code}，实际为 {sorted(codes)}")
    return data


def load_data1(method, instruments, task_id, period, features, regime,
               ret_name):
    """读取上游已经切好的 train/val；不在调用端重新切割。"""
    data_dir = os.path.join(base_path, method, instruments, "temp", "model",
                            str(task_id), str(period), "rl", "data")
    train_path = os.path.join(data_dir, "train_data.feather")
    val_path = os.path.join(data_dir, "val_data.feather")
    train = pd.read_feather(train_path)
    val = pd.read_feather(val_path)
    required = {"trade_time", "code", ret_name, *features, *regime}
    for path, frame in ((train_path, train), (val_path, val)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    train = train.rename(columns={ret_name: "nxt1_ret"})
    val = val.rename(columns={ret_name: "nxt1_ret"})
    columns = ["trade_time", "code", "nxt1_ret"] + features + regime
    train, val = train[columns].copy(), val[columns].copy()
    for frame in (train, val):
        frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="raise")
    train = _sanitize_frame(
        train.sort_values("trade_time").reset_index(drop=True), features + regime)
    val = _sanitize_frame(
        val.sort_values("trade_time").reset_index(drop=True), features + regime)
    train_codes = set(train["code"].astype(str))
    val_codes = set(val["code"].astype(str))
    if len(train_codes) != 1 or train_codes != val_codes:
        raise ValueError(
            f"train/val 必须是同一单品种，实际 train={sorted(train_codes)} "
            f"val={sorted(val_codes)}")
    expected_code = next(iter(train_codes))
    if train["trade_time"].max() >= val["trade_time"].min():
        raise ValueError(f"{expected_code} train/val 时间重叠或顺序错误")
    return train, val


def load_test_data(method, instruments, task_id, period, features, regime,
                   ret_name, expected_code):
    """保持 6.1.4 的测试集调用接口。"""
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
    return load_rl_params(file_dirs=file_dirs, trade_id=trade_id,
                          model_id=model_id, feature_id=feature_id,
                          env_id=env_id, train_id=train_id,
                          regime_id=regime_id)


def _run_identity(trade_params, env_params, model_params, train_params,
                  selected_features, min_regime, daily_regime):
    total = copy.deepcopy(trade_params)
    total.update(env_params)
    total.update(model_params)
    total.update(train_params)
    total.update(selected_features=selected_features,
                 min_regime=min_regime, daily_regime=daily_regime)
    return Params.create_tag(total)


def _validate_core(period, trade_params, model_params):
    if int(period) != 5 or trade_params["ret_name"] != "nxt1_ret_5h":
        raise ValueError("rl016 当前要求 period=5、ret_name=nxt1_ret_5h")
    if abs(float(model_params.get("gamma", 0.0))) > 1e-12:
        raise ValueError("rl016 是完整5分钟标签的单步MSE拟合，gamma 必须为 0.0")


def train(method, instruments, task_id, period, env_id, trade_id, model_id,
          train_id, feature_id, regime_id):
    task_key = str(task_id)
    if task_key not in PAIRE_TASK:
        raise KeyError(f"未配置主任务 {task_key} 的配对品种")
    (env_params, trade_params, model_params, train_params, selected_features,
     min_regime, daily_regime) = _load_params(
         method, instruments, task_id, period, env_id, trade_id, model_id,
         train_id, feature_id, regime_id)
    _validate_core(period, trade_params, model_params)

    name = _run_identity(trade_params, env_params, model_params, train_params,
                         selected_features, min_regime, daily_regime)
    output_dir = os.path.join(base_path, method, instruments, "temp", "model",
                              str(task_id), str(period), "rl", "result", str(name))
    tensorboard_dir = os.path.join(tensorboard_path, instruments, str(task_id),
                                   str(name))
    os.makedirs(output_dir, exist_ok=True)
    logger.configure(log_file=os.path.join(output_dir, "model.log"))

    right_instruments, right_task = PAIRE_TASK[task_key]
    rb_train, rb_val = load_data1(
        method=method, instruments=instruments, task_id=task_id, period=period,
        features=selected_features, regime=min_regime,
        ret_name=trade_params["ret_name"])
    hc_train, hc_val = load_data1(
        method=method, instruments=right_instruments, task_id=right_task,
        period=period, features=selected_features, regime=min_regime,
        ret_name=trade_params["ret_name"])
    if set(rb_train["code"].astype(str)) != {"RB"}:
        raise ValueError("主任务训练集必须是 RB")
    if set(hc_train["code"].astype(str)) != {"HC"}:
        raise ValueError("配对任务训练集必须是 HC")
    train_data = _merge_assets(rb_train, hc_train)
    val_data = _merge_assets(rb_val, hc_val)

    env_config = {
        "holding_period": int(env_params["holding_period"]),
        "reward_scale": float(env_params["reward_scale"]),
        "prediction_bound_std": float(env_params.get("prediction_bound_std", 3.0)),
        "use_tcn": bool(env_params["use_tcn"]),
        "default_lookback": int(env_params["default_lookback"]),
        "asset_sampling": "balanced",
        "max_episode_steps": int(env_params["max_episode_steps"]),
        "train_scheme": env_params["train_scheme"],
        "seed": int(env_params["seed"]),
    }
    sac_config = {
        "learning_rate": model_params["learning_rate"],
        "buffer_size": model_params["buffer_size"],
        "learning_starts": model_params["learning_starts"],
        "batch_size": model_params["batch_size"],
        "tau": model_params["tau"],
        "gamma": model_params["gamma"],
        "train_freq": model_params["train_freq"],
        "gradient_steps": model_params["gradient_steps"],
        "ent_coef": model_params["ent_coef"],
        "target_update_interval": model_params["target_update_interval"],
        "policy_kwargs": copy.deepcopy(model_params["policy_kwargs"]),
    }
    signal_config = {}
    logger.info(f"训练集: {len(train_data)} 行，RB={len(rb_train)} HC={len(hc_train)}")
    logger.info(f"校验集: {len(val_data)} 行，RB={len(rb_val)} HC={len(hc_val)}")
    logger.info(f"env_config: {env_config}")
    logger.info(f"sac_config: {sac_config}")
    logger.info(f"signal_config: {signal_config}")
    logger.info(f"train_params: {train_params}")
    logger.info(f"env_params: {env_params}")
    logger.info(f"trade_params: {trade_params}")
    logger.info(f"model_params: {model_params}")
    logger.info(f"selected_features: {selected_features}")
    logger.info(f"min_regime: {min_regime}")
    logger.info(f"daily_regime: {daily_regime}")

    model, training_info = train_model(
        train_df=train_data, val_df=val_data,
        features=selected_features + min_regime,
        env_config=env_config, sac_config=sac_config,
        signal_config=signal_config,
        output_dir=output_dir, tensorboard_dir=tensorboard_dir,
        total_timesteps=int(train_params["total_timesteps"]),
        eval_freq=int(train_params["eval_freq"]),
        full_eval_freq=int(train_params.get("full_eval_freq", 50000)),
        save_freq=int(train_params["save_freq"]),
        val_window_steps=int(train_params.get("val_window_steps", 60)),
        val_windows_per_asset=int(train_params.get("val_windows_per_asset", 160)),
        validation_batch_size=int(train_params.get("validation_batch_size", 4096)),
        ic_resampling_minutes=int(
            train_params.get("ic_resampling_minutes", 5)),
        ic_roll_window=int(train_params.get("ic_roll_window", 15)),
        ic_min_periods=int(train_params.get("ic_min_periods", 5)),
        min_valid_rolling_ic=int(
            train_params.get("min_valid_rolling_ic", 20)),
        early_stop_patience_evals=int(train_params["early_stop_patience_evals"]),
        early_stop_min_evals=int(train_params["early_stop_min_evals"]),
        early_stop_min_delta=float(train_params.get("early_stop_ic_min_delta", 0.0)),
        early_stop_start_timesteps=int(train_params["early_stop_start_timesteps"]),
        verbose=1)
    return model, training_info


def predict(method, instruments, task_id, period, env_id, trade_id, model_id,
            train_id, feature_id, regime_id):
    task_key = str(task_id)
    if task_key not in PAIRE_TASK:
        raise KeyError(f"未配置主任务 {task_key} 的配对品种")
    (env_params, trade_params, model_params, train_params, selected_features,
     min_regime, daily_regime) = _load_params(
         method, instruments, task_id, period, env_id, trade_id, model_id,
         train_id, feature_id, regime_id)
    _validate_core(period, trade_params, model_params)
    name = _run_identity(trade_params, env_params, model_params, train_params,
                         selected_features, min_regime, daily_regime)
    output_dir = os.path.join(base_path, method, instruments, "temp", "model",
                              str(task_id), str(period), "rl", "result", str(name))
    right_instruments, right_task = PAIRE_TASK[task_key]
    rb = load_test_data(
        method=method, instruments=instruments, task_id=task_id,
        period=period, features=selected_features, regime=min_regime,
        ret_name=trade_params["ret_name"], expected_code="RB")
    hc = load_test_data(
        method=method, instruments=right_instruments, task_id=right_task,
        period=period, features=selected_features, regime=min_regime,
        ret_name=trade_params["ret_name"], expected_code="HC")
    best_model = os.path.join(output_dir, "models", "best_model",
                              "best_model")
    config_path = os.path.join(output_dir, "config.json")
    if not (os.path.isfile(best_model) or os.path.isfile(best_model + ".zip")):
        raise FileNotFoundError(f"IC 最佳模型不存在: {best_model}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"训练配置不存在: {config_path}")
    test_data = _merge_assets(rb, hc)
    output_path = os.path.join(output_dir, "metrics", "test_results.csv")
    print(f"[TEST_START] RB={len(rb)} HC={len(hc)} model=best_ic")
    result = predict_test_set(
        model_path=best_model, config_path=config_path, test_df=test_data,
        output_path=output_path, deterministic=True)
    print(f"[TEST_END] rows={len(result)} output={output_path}")
    return result


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
