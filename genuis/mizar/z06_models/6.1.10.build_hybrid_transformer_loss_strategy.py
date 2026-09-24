"""HybridTransformer-Loss 动态多资产调用端。

资产、task id 和 instrument 由 ``asset_specs`` 显式传入，代码中不维护
RB/HC、P/Y或其他品种的固定映射。
"""

import copy
import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.uvx import *
from lib.nn005.forecast import find_model_files, compare_seeds
from lib.nn005.predict import predict_test_set
from lib.nn005.train import train_model


def _parse_mapping(value, name):
    """允许Python对象或YAML对象直接传入，也支持命令行JSON字符串。"""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}不是合法JSON: {exc}") from exc
    return value


def _normalize_asset_specs(asset_specs, instruments=None, task_id=None):
    """规范化资产配置，并校验code、instrument、task id均唯一。

    推荐输入：
    [
        {"instruments": "p_instrument", "task_id": "p_task", "code": "P"},
        {"instruments": "y_instrument", "task_id": "y_task", "code": "Y"},
    ]

    ``code`` 可省略，此时从 ``INSTRUMENTS_CODES`` 推导。为兼容已有调用，
    当asset_specs为空时只会使用主调用的instruments/task_id；训练和融合仍要求
    至少两个资产，因此不会静默退化成单资产训练。
    """
    asset_specs = _parse_mapping(asset_specs, "asset_specs")
    if asset_specs is None:
        if instruments is None or task_id is None:
            raise ValueError("asset_specs为空时必须提供instruments和task_id")
        asset_specs = [{"instruments": instruments, "task_id": task_id}]
    if not isinstance(asset_specs, (list, tuple)) or not asset_specs:
        raise ValueError("asset_specs必须是非空列表")

    normalized = []
    for index, raw in enumerate(asset_specs):
        if not isinstance(raw, dict):
            raise TypeError(f"asset_specs[{index}]必须是字典")
        asset_instruments = str(raw.get("instruments", "")).strip()
        asset_task_id = str(raw.get("task_id", "")).strip()
        if not asset_instruments or not asset_task_id:
            raise ValueError(f"asset_specs[{index}]必须包含instruments和task_id")
        code = raw.get("code")
        if code is None:
            if asset_instruments not in INSTRUMENTS_CODES:
                raise KeyError(
                    f"INSTRUMENTS_CODES中没有{asset_instruments}，请显式传入code")
            code = INSTRUMENTS_CODES[asset_instruments]
        code = str(code).strip().upper()
        if not code:
            raise ValueError(f"asset_specs[{index}].code不能为空")
        normalized.append({
            "instruments": asset_instruments,
            "task_id": asset_task_id,
            "code": code,
        })

    if len(normalized) < 2:
        raise ValueError("动态多资产训练至少需要两个asset_specs")
    for key in ("instruments", "task_id", "code"):
        values = [item[key] for item in normalized]
        if len(values) != len(set(values)):
            raise ValueError(f"asset_specs中的{key}不能重复: {values}")
    if instruments is not None and normalized[0]["instruments"] != str(
            instruments):
        raise ValueError("asset_specs第一项必须与主调用instruments一致")
    if task_id is not None and normalized[0]["task_id"] != str(task_id):
        raise ValueError("asset_specs第一项必须与主调用task_id一致")
    return normalized


def _normalize_seed_run_ids(seed_run_ids):
    seed_run_ids = _parse_mapping(seed_run_ids, "seed_run_ids")
    if not isinstance(seed_run_ids, dict):
        raise TypeError("seed_run_ids必须是{seed: run_id}字典")
    result = {int(seed): str(run_id) for seed, run_id in seed_run_ids.items()}
    if len(result) != len(seed_run_ids):
        raise ValueError("seed转换为整数后存在重复值")
    if len(result) < 2:
        raise ValueError("forecast至少需要两个不同seed的模型")
    return result


def _sanitize_frame(df, columns):
    columns = [column for column in columns if column in df.columns]
    if not columns:
        return df
    df = df.copy()
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(df[columns].to_numpy(dtype=np.float64))
    if int(bad.sum()):
        print(f"[WARN] 数据中发现 {int(bad.sum())} 个 NaN/Inf，已填充为 0.0")
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _load_split(method, instruments, task_id, period, trial_id, split,
                features, regime, ret_name, expected_code):
    data_dir = os.path.join(base_path, method, instruments, "temp", "model",
                            str(task_id), str(period), "rl", str(trial_id),
                            "data")
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


def load_data1(method, instruments, task_id, period, trial_id, features,
               regime, ret_name, expected_code):
    train = _load_split(method, instruments, task_id, period, trial_id,
                        "train", features, regime, ret_name, expected_code)
    val = _load_split(method, instruments, task_id, period, trial_id, "val",
                      features, regime, ret_name, expected_code)
    if train["trade_time"].max() >= val["trade_time"].min():
        raise ValueError(f"{expected_code} train/val 时间重叠或顺序错误")
    return train, val


def load_test_data(method, instruments, task_id, period, features, regime,
                   ret_name, expected_code):
    return _load_split(method, instruments, task_id, period, "test", features,
                       regime, ret_name, expected_code)


def _merge_assets(*frames):
    if len(frames) < 2:
        raise ValueError("至少需要两个资产数据集")
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["code", "trade_time"]).any():
        raise ValueError("联合数据存在重复的 code/trade_time")
    return data.sort_values(["code", "trade_time"]).reset_index(drop=True)


def _load_params(method, instruments, task_id, period, trial_id, env_id,
                 trade_id, model_id, train_id, feature_id, regime_id):
    file_dirs = os.path.join(base_path, method, instruments, "temp", "model",
                             str(task_id), str(period), "rl", str(trial_id))
    return load_rl_params(file_dirs=file_dirs,
                          trade_id=trade_id,
                          model_id=model_id,
                          feature_id=feature_id,
                          env_id=env_id,
                          train_id=train_id,
                          regime_id=regime_id,
                          name='mtf')


def _run_identity(trade_params, env_params, model_params, train_params,
                  selected_features, min_regime, daily_regime, asset_specs):
    total = {"algorithm": "hybrid_transformer_loss_v2_dynamic_assets"}
    for values in (trade_params, env_params, model_params, train_params):
        total.update(copy.deepcopy(values))
    total.update(selected_features=selected_features,
                 min_regime=min_regime,
                 daily_regime=daily_regime,
                 asset_specs=copy.deepcopy(asset_specs))
    return Params.create_tag(total)


def _validate_core(period, trade_params, env_params, train_params):
    if int(period) != 5 or trade_params["ret_name"] != "nxt1_ret_5h":
        raise ValueError(
            "HybridTransformer-Loss 要求 period=5、ret_name=nxt1_ret_5h")
    if not bool(env_params.get("use_hybrid_transformer", True)):
        raise ValueError(
            "HybridTransformer-Loss 要求 use_hybrid_transformer=true")
    required_loss = {"loss_type", "huber_delta", "corr_weight"}
    missing = required_loss - set(train_params)
    if missing:
        raise ValueError(f"损失参数缺少: {sorted(missing)}")


def _result_dir(method, instruments, task_id, period, trial_id, name):
    return os.path.join(base_path, method, instruments, "temp", "model",
                        str(task_id), str(period), "rl", str(trial_id),
                        "hybrid_transformer_loss", "result", str(name))


def train(method, instruments, task_id, period, trial_id, env_id, trade_id,
          model_id, train_id, feature_id, regime_id, asset_specs):
    pdb.set_trace()
    assets = _normalize_asset_specs(asset_specs, instruments, task_id)
    (env_params, trade_params, model_params, train_params, selected_features,
     min_regime, daily_regime) = _load_params(method=method,
                                              instruments=instruments,
                                              task_id=task_id,
                                              trial_id=trial_id,
                                              period=period,
                                              env_id=env_id,
                                              trade_id=trade_id,
                                              model_id=model_id,
                                              train_id=train_id,
                                              feature_id=feature_id,
                                              regime_id=regime_id)
    _validate_core(period, trade_params, env_params, train_params)
    name = _run_identity(trade_params, env_params, model_params, train_params,
                         selected_features, min_regime, daily_regime, assets)
    output_dir = _result_dir(method=method,
                             instruments=instruments,
                             task_id=task_id,
                             period=period,
                             trial_id=trial_id,
                             name=name)
    os.makedirs(output_dir, exist_ok=True)
    logger.configure(log_file=os.path.join(output_dir, "model.log"))

    train_frames, val_frames = [], []
    pdb.set_trace()
    for asset in assets:
        asset_train, asset_val = load_data1(method, asset["instruments"],
                                            asset["task_id"], period, trial_id,
                                            selected_features, min_regime,
                                            trade_params["ret_name"],
                                            asset["code"])
        train_frames.append(asset_train)
        val_frames.append(asset_val)
    train_data = _merge_assets(*train_frames)
    val_data = _merge_assets(*val_frames)
    env_config = {
        "holding_period": int(env_params["holding_period"]),
        "prediction_bound_std":
        float(env_params.get("prediction_bound_std", 3.0)),
        "use_hybrid_transformer": True,
        "default_lookback": int(env_params.get("default_lookback", 20)),
        "asset_sampling": "balanced",
        "seed": int(env_params.get("seed", 42)),
    }
    train_counts = {
        asset["code"]: len(frame)
        for asset, frame in zip(assets, train_frames)
    }
    val_counts = {
        asset["code"]: len(frame)
        for asset, frame in zip(assets, val_frames)
    }
    logger.info(f"HybridTransformer-Loss资产配置: {assets}")
    logger.info(f"HybridTransformer-Loss训练集: {len(train_data)} 行，"
                f"分品种={train_counts}")
    logger.info(f"HybridTransformer-Loss校验集: {len(val_data)} 行，"
                f"分品种={val_counts}")
    logger.info(f"env_config: {env_config}")
    logger.info(f"model_params: {model_params}")
    logger.info(f"train_params: {train_params}")
    logger.info(f"selected_features: {selected_features}")
    return train_model(train_data, val_data, selected_features + min_regime,
                       env_config, copy.deepcopy(model_params),
                       copy.deepcopy(train_params), output_dir)


def predict(method, instruments, task_id, period, trial_id, env_id, trade_id,
            model_id, train_id, feature_id, regime_id, asset_specs):
    assets = _normalize_asset_specs(asset_specs, instruments, task_id)
    (env_params, trade_params, model_params, train_params, selected_features,
     min_regime, daily_regime) = _load_params(method, instruments, task_id,
                                              period, trial_id, env_id,
                                              trade_id, model_id, train_id,
                                              feature_id, regime_id)
    _validate_core(period, trade_params, env_params, train_params)
    name = _run_identity(trade_params, env_params, model_params, train_params,
                         selected_features, min_regime, daily_regime, assets)
    output_dir = _result_dir(method, instruments, task_id, period, trial_id,
                             name)
    test_frames = [
        load_test_data(method, asset["instruments"], asset["task_id"], period,
                       selected_features, min_regime, trade_params["ret_name"],
                       asset["code"]) for asset in assets
    ]
    model_path = os.path.join(output_dir, "models", "best_model.pt")
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"HybridTransformer-Loss 最佳模型不存在: {model_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"HybridTransformer-Loss 训练配置不存在: {config_path}")
    output_path = os.path.join(output_dir, "metrics", "test_results.csv")
    counts = {
        asset["code"]: len(frame)
        for asset, frame in zip(assets, test_frames)
    }
    print(f"[HYBRID_TRANSFORMER_LOSS_TEST_START] assets={counts}")
    return predict_test_set(model_path,
                            config_path,
                            _merge_assets(*test_frames),
                            output_path,
                            inference_batch_size=int(
                                train_params.get("inference_batch_size",
                                                 4096)),
                            device=str(model_params.get("device", "auto")))


def forecast(method,
             instruments,
             task_id,
             period,
             trial_id,
             seed_run_ids,
             asset_specs,
             split="val",
             output_dir=None,
             inference_batch_size=4096,
             device="auto"):
    """复用多个已训练的纯Huber模型，预测单模型及等权融合。

    ``seed_run_ids`` 为 ``{seed: run_id}``；``asset_specs`` 与训练时完全
    一致。两者均可传Python/YAML对象，也可传JSON字符串。
    split仅支持val/test，自动读取对应已切割数据；不读取当前YAML来
    重新推导训练ID，特征、收益尺度和评价参数以各训练目录配置为准。
    三个seed时返回三个单模型及一个融合预测；同时返回统一评价结果。
    输出到独立新目录，不覆盖原单模型结果，也不自动挑选测试集赢家。
    """
    assets = _normalize_asset_specs(asset_specs, instruments, task_id)
    ids = _normalize_seed_run_ids(seed_run_ids)
    run_dirs = {
        seed:
        find_model_files(method, instruments, task_id, period, trial_id,
                         run_id)["run_dir"]
        for seed, run_id in ids.items()
    }
    asset_paths = {
        asset["code"]:
        os.path.join(base_path, method, asset["instruments"], "temp", "model",
                     str(asset["task_id"]), str(period), "rl", str(trial_id),
                     "data", f"{split}_data.feather")
        for asset in assets
    }
    if output_dir is None:
        #name = "ensemble_" + "_".join(f"s{seed}_{ids[seed]}"
        #                              for seed in sorted(ids))
        name = "_".join(f"{ids[seed]}" for seed in sorted(ids))
        output_dir = os.path.join(
            _result_dir(method, instruments, task_id, period, trial_id,
                        os.path.join("ensemble", name)), split)

    return compare_seeds(run_dirs=run_dirs,
                         asset_paths=asset_paths,
                         output_dir=output_dir,
                         split=split,
                         batch_size=inference_batch_size,
                         device=device,
                         ret_name="nxt1_ret_{0}h".format(period),
                         return_predictions=True)


def _required_variant(variant, name):
    value = getattr(variant, name, None)
    if value is None:
        raise ValueError(f"缺少{name}；请在YAML或直接函数调用中显式传入")
    return value


def _cli_parser():
    """在原Tactix参数上增加动态资产和融合模型参数。

    一般由YAML直接提供list/dict；命令行传入时使用JSON。
    """
    parser = Tactix().params()
    parser.add_argument("--asset_specs",
                        type=json.loads,
                        default=None,
                        help=("JSON资产列表，例如 "
                              "'[{\"instruments\":\"pbb\",\"task_id\":\"...\","
                              "\"code\":\"P\"},{\"instruments\":\"ybb\","
                              "\"task_id\":\"...\",\"code\":\"Y\"}]'"))
    parser.add_argument(
        "--seed_run_ids",
        type=json.loads,
        default=None,
        help='forecast使用的JSON字典，例如 \'{"42":"run1","2026":"run2"}\'')
    return parser


if __name__ == "__main__":
    variant = Tactix().start(callback=_cli_parser)

    if variant.form == "train":
        train(method=variant.method,
              instruments=variant.instruments,
              task_id=variant.task_id,
              period=variant.period,
              env_id=variant.env_id,
              trade_id=variant.trade_id,
              model_id=variant.model_id,
              train_id=variant.train_id,
              feature_id=variant.feature_id,
              regime_id=variant.regime_id,
              trial_id=variant.trial_id,
              asset_specs=_required_variant(variant, "asset_specs"))
    elif variant.form == "predict":
        predict(method=variant.method,
                instruments=variant.instruments,
                task_id=variant.task_id,
                period=variant.period,
                env_id=variant.env_id,
                trade_id=variant.trade_id,
                model_id=variant.model_id,
                train_id=variant.train_id,
                feature_id=variant.feature_id,
                regime_id=variant.regime_id,
                trial_id=variant.trial_id,
                asset_specs=_required_variant(variant, "asset_specs"))
    elif variant.form == 'forecast':
        result = forecast(
            method=variant.method,
            instruments=variant.instruments,
            task_id=variant.task_id,
            period=variant.period,
            seed_run_ids=_required_variant(variant, "seed_run_ids"),
            asset_specs=_required_variant(variant, "asset_specs"),
            split=variant.split,  #"val",
            inference_batch_size=variant.inference_batch_size,
            device=variant.device,
            trial_id=variant.trial_id)
    else:
        raise ValueError(f"不支持 form={variant.form}")
