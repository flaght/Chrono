import os, copy, json, pdb
import pandas as pd
import numpy as np
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

from kdutils.tactix import Tactix
from kdutils.macro2 import base_path
from kdutils.logger import logger
from kdutils.macro2 import *

from lib.uvx import *
from lib.utils.params import Params

from lib.rl023.train import train_model
from lib.rl023.predict import predict_test_set
from lib.rl023.signal import Config
from lib.rl023.custom_feature import CrossSectionalExtractor

extractor_mapping = {
    "CrossSectionalExtractor": CrossSectionalExtractor,
}


def require_keys(section_name: str, data: Dict, keys: List[str]):
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"{section_name} 缺少必要参数: {missing}")


def load_data_train_val(method, task_id, features, ret_name):
    target_dir = os.path.join(base_path, method, "rl", str(task_id))
    train_data = pd.read_feather(
        os.path.join(target_dir, "derive_train_data.feather"))
    val_data = pd.read_feather(
        os.path.join(target_dir, "derive_val_data.feather"))

    train_data.rename(columns={ret_name: "nxt1_ret"}, inplace=True)
    val_data.rename(columns={ret_name: "nxt1_ret"}, inplace=True)
    pdb.set_trace()
    train_data = train_data[["trade_time", "code", "nxt1_ret"] + features]
    val_data = val_data[["trade_time", "code", "nxt1_ret"] + features]

    train_data = train_data.sort_values(["trade_time",
                                         "code"]).reset_index(drop=True)
    val_data = val_data.sort_values(["trade_time",
                                     "code"]).reset_index(drop=True)
    return train_data, val_data


def load_data_test(method, task_id, features, ret_name):
    target_dir = os.path.join(base_path, method, "rl", str(task_id))
    test_data = pd.read_feather(
        os.path.join(target_dir, "derive_test_data.feather"))

    test_data.rename(columns={ret_name: "nxt1_ret"}, inplace=True)
    test_data = test_data[["trade_time", "code", "nxt1_ret"] + features]
    test_data = test_data.sort_values(["trade_time",
                                       "code"]).reset_index(drop=True)
    return test_data


def build_sac_config(model_params, n_features, subset_size, use_custom_policy):
    sac_config = {
        "learning_rate": float(model_params["learning_rate"]),
        "buffer_size": model_params["buffer_size"],
        "learning_starts": model_params["learning_starts"],
        "batch_size": model_params["batch_size"],
        "tau": model_params["tau"],
        "gamma": model_params["gamma"],
        "train_freq": model_params["train_freq"],
        "gradient_steps": model_params["gradient_steps"],
        "ent_coef": model_params["ent_coef"],
        "target_update_interval": model_params["target_update_interval"],
    }
    if use_custom_policy:
        policy_kwargs = {
            "n_assets": subset_size,
            "n_stock_features": n_features,
        }
        if "hidden_dim" in model_params:
            policy_kwargs["hidden_dim"] = model_params["hidden_dim"]
        sac_config["policy_kwargs"] = policy_kwargs

    else:
        if "policy_kwargs" in model_params:
            policy_kwargs = model_params["policy_kwargs"]
            extractor_kwargs = policy_kwargs["features_extractor_kwargs"]
            net_arch = policy_kwargs["net_arch"]
            sac_config["policy_kwargs"] = {
                "features_extractor_class":
                extractor_mapping[policy_kwargs["features_extractor_class"]],
                "features_extractor_kwargs": {
                    "features_dim": extractor_kwargs["features_dim"],
                    "n_assets": subset_size,
                    "n_stock_features": n_features,
                    "stock_encoder_mid_dim":
                    extractor_kwargs["encoder_mid_dim"],
                    "stock_encoder_out_dim":
                    extractor_kwargs["encoder_out_dim"],
                },
                "net_arch": {
                    "pi": net_arch["pi"],
                    "qf": net_arch["qf"],
                },
            }

    return sac_config


def train(method, task_id, env_id, trade_id, model_id, train_id, feature_id):
    file_dirs = os.path.join(base_path, method, "temp", "trl", task_id)
    pdb.set_trace()
    env_params, trade_params, model_params, train_params, selected_features = load_rl_params(
        file_dirs=file_dirs,
        trade_id=trade_id,
        model_id=model_id,
        feature_id=feature_id,
        env_id=env_id,
        train_id=train_id,
    )

    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({"selected_features": selected_features})
    name = Params.create_tag(total_params)

    output_dir = os.path.join(base_path, method, "temp", "trl", str(task_id),
                              str(name))

    os.makedirs(output_dir, exist_ok=True)
    logger.configure(log_file=os.path.join(output_dir, "model.log"))

    train_data, val_data = load_data_train_val(
        method=method,
        task_id=task_id,
        ret_name=trade_params["ret_name"],
        features=selected_features,
    )

    return_columns = (train_data.filter(regex="^nxt1").columns.to_list() +
                      train_data.filter(regex="^abret_").columns.to_list())
    features = [
        f for f in train_data.columns
        if f not in ["trade_time", "code"] + return_columns
    ]
    train_data = train_data[["trade_time", "code", "nxt1_ret"] + features]
    val_data = val_data[["trade_time", "code", "nxt1_ret"] + features]

    env_config = {
        "subset_size": env_params["subset_size"],
        "episode_len": env_params["episode_len"],
        "reward_scale": env_params["reward_scale"],
        "ic_scale": env_params["ic_scale"],
        "negative_ic_penalty": env_params["negative_ic_penalty"],
        "reward_mode": env_params["reward_mode"],
        "reward_top_k": env_params["reward_top_k"],
        "seed": 42
    }

    signal_config = Config(
        min_weight=trade_params["min_weight"],
        max_weight=trade_params["max_weight"],
        normalize=trade_params["normalize"],
        top_k=trade_params["top_k"],
        cost_rate=trade_params["cost_rate"],
        turnover_penalty=trade_params["turnover_penalty"],
        rebalance_window=trade_params["rebalance_window"],
        softmax_temperature=trade_params["softmax_temperature"],
    )

    use_custom_policy = bool(model_params["use_custom_policy"])

    sac_config = build_sac_config(
        model_params=model_params,
        n_features=len(features),
        subset_size=env_config["subset_size"],
        use_custom_policy=use_custom_policy,
    )

    pdb.set_trace()
    logger.info(f"训练集: {len(train_data)} 行")
    logger.info(f"校验集: {len(val_data)} 行")
    logger.info(f"features: {len(features)}")
    logger.info(f"env_config: {env_config}")
    logger.info(f"signal_config: {signal_config}")
    logger.info(f"sac_config: {sac_config}")
    logger.info(f"use_custom_policy: {use_custom_policy}")

    model, training_info = train_model(
        train_df=train_data,
        val_df=val_data,
        features=features,
        env_config=env_config,
        sac_config=sac_config,
        signal_config=signal_config,
        output_dir=output_dir,
        total_timesteps=train_params["total_timesteps"],
        eval_freq=train_params["eval_freq"],
        eval_n_episodes=train_params["eval_n_episodes"],
        save_freq=train_params["save_freq"],
        verbose=1,
        use_custom_policy=use_custom_policy,
    )
    logger.info(f"训练完成: {training_info}")

    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(total_params, f, indent=2, default=str)


def predict(method, task_id, env_id, trade_id, model_id, train_id, feature_id):

    new_ret_name = 'nxt1_ret_1h'
    top_k = 20

    pdb.set_trace()
    file_dirs = os.path.join(base_path, method, "temp", "trl", task_id)
    env_params, trade_params, model_params, train_params, selected_features = load_rl_params(
        file_dirs=file_dirs,
        trade_id=trade_id,
        model_id=model_id,
        feature_id=feature_id,
        env_id=env_id,
        train_id=train_id,
    )

    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({"selected_features": selected_features})
    name = Params.create_tag(total_params)

    output_dir = os.path.join(base_path, method, "temp", "trl", str(task_id),
                              str(name))

    test_data = load_data_test(
        method=method,
        task_id=task_id,
        ret_name=new_ret_name,  ## 预测 #trade_params["ret_name"],
        features=selected_features,
    )
    best_model_zip = os.path.join(output_dir, "models", "best_model",
                                  "best_model.zip")
    best_model = os.path.join(output_dir, "models", "best_model", "best_model")
    model_path = best_model_zip if os.path.exists(
        best_model_zip) else best_model
    config_path = os.path.join(output_dir, "config.json")

    signals_df = predict_test_set(model_path=model_path,
                                  config_path=config_path,
                                  test_df=test_data,
                                  top_k=top_k,
                                  output_path=os.path.join(
                                      output_dir, "metrics",
                                      "results_{0}_{1}.csv".format(
                                          str(new_ret_name), str(top_k))),
                                  deterministic=True,
                                  return_details=False)

    print(f"预测完成，共 {len(signals_df)} 个时间步")
    if 'n_holdings' in signals_df.columns:
        print(f"平均持仓数量: {signals_df['n_holdings'].mean():.1f}")
    if 'turnover' in signals_df.columns:
        print(f"平均换手率: {signals_df['turnover'].mean():.6f}")
    if 'rank_ic' in signals_df.columns:
        print(f"平均 Rank IC: {signals_df['rank_ic'].mean():.6f}")
    print(f"手续费费率: {trade_params['cost_rate']}")


if __name__ == "__main__":
    variant = Tactix().start()
    predict(method=variant.method,
          task_id=variant.task_id,
          trade_id=variant.trade_id,
          env_id=variant.env_id,
          train_id=variant.train_id,
          model_id=variant.model_id,
          feature_id=variant.feature_id)
