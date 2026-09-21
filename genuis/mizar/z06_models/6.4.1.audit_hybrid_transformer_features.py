import copy, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.uvx import *
from lib.nn004.correlation import feature_correlation

PAIRE_TASK = {"113001": ("hcb", "134001")}
OLD_FEATURE_COUNT = 24


def _run_identity(trade_params, env_params, model_params, train_params,
                  selected_features, min_regime, daily_regime):
    total = {"algorithm": "hybrid_transformer_loss_v1"}
    for values in (trade_params, env_params, model_params, train_params):
        total.update(copy.deepcopy(values))
    total.update(selected_features=selected_features,
                 min_regime=min_regime,
                 daily_regime=daily_regime)
    return Params.create_tag(total)


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
    # 审计必须保留原始缺失状态，不能复用训练端 NaN/Inf -> 0 的清洗。
    data = data.sort_values("trade_time").reset_index(drop=True)

    codes = set(data["code"].astype(str))
    if data.empty or codes != {expected_code}:
        raise ValueError(f"{path} 应只包含 {expected_code}，实际为 {sorted(codes)}")
    return data


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


def run(method,
        instruments,
        task_id,
        period,
        trial_id,
        env_id,
        trade_id,
        model_id,
        train_id,
        feature_id,
        regime_id,
        old_feature_count=OLD_FEATURE_COUNT,
        strong_threshold=0.85,
        duplicate_threshold=0.95):
    task_key = str(task_id)
    if task_key not in PAIRE_TASK:
        raise KeyError(f"未配置主任务 {task_key} 的配对品种")
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
    old_features = [
        "MDIFF(5,EMA(60,MRANK(15,'dv002_1_2_1')))",
        "EMA(60,MRANK(15,'dv002_1_2_1'))", "MCPS(120,'dv002_2_3_1')",
        "MDIFF(30,MCPS(120,'dv002_2_3_1'))",
        "MPERCENT(60,SIGLOG2ABS('dv002_2_3_1'))",
        "SIGLOG10ABS(MQUANTILE(240,'dv002_2_3_1'))",
        "MDIFF(5,MADiff(90,MRANK(240,MADiff(90,'dv002_2_3_1'))))",
        "MMinDiff(90,SIGLOG10ABS(MQUANTILE(90,'iv012_1_2_1')))",
        "SIGLOG10ABS(MPERCENT(60,MMinDiff(120,'dv002_2_3_1')))",
        "MQUANTILE(240,SIGLOG10ABS('dv002_2_3_1'))",
        "MRes(120,'cj007_10_15_1',MPERCENT(30,'iv012_2_3_1'))",
        "MPERCENT(120,MADiff(240,'dv002_2_3_1'))",
        "MCPS(120,MDIFF(10,MMaxDiff(60,MCPS(120,'dv002_2_3_1'))))",
        "TANH(MQUANTILE(120,SIGLOG2ABS('dv002_2_3_1')))",
        "MCPS(120,MDIFF(10,'dv002_2_3_1'))", "ASIN(MCPS(120,'dv002_2_3_1'))",
        "MRANK(120,MDIFF(5,SIGLOG2ABS('dv002_2_3_1')))",
        "MPERCENT(120,MQUANTILE(90,'dv002_2_3_1'))",
        "SIGLOG2ABS(MRANK(60,SIGLOG10ABS('iv012_1_2_1')))",
        "MRANK(240,'iv012_1_2_1')", "MPERCENT(120,MMinDiff(90,'tv018_2_3_1'))",
        "MPERCENT(120,MMinDiff(90,MMinDiff(90,'tv018_2_3_1')))",
        "MQUANTILE(90,MQUANTILE(90,'iv012_1_2_1'))",
        "MPERCENT(120,MPERCENT(120,MPERCENT(120,MMinDiff(90,'tv018_2_3_1'))))"
    ]
    old_feature_count = int(old_feature_count)
    if old_feature_count != len(old_features):
        raise ValueError(f"old_feature_count={old_feature_count} 与固化旧特征数"
                         f" {len(old_features)} 不一致")
    missing_old = set(old_features) - set(selected_features)
    if missing_old:
        raise ValueError(f"当前配置缺少固化旧特征: {sorted(missing_old)}")
    if len(set(selected_features)) != len(selected_features):
        raise ValueError("selected_features 存在重复名称")
    old_set = set(old_features)
    new_features = [feature for feature in selected_features
                    if feature not in old_set]
    if not new_features:
        raise ValueError("当前配置没有新增特征")

    right_instruments, right_task = PAIRE_TASK[task_key]

    left_train = _load_split(method,
                             instruments,
                             task_id,
                             period,
                             trial_id=trial_id,
                             split="train",
                             features=selected_features,
                             regime=min_regime,
                             ret_name=trade_params["ret_name"],
                             expected_code="RB")

    right_train = _load_split(method,
                              right_instruments,
                              right_task,
                              period,
                              trial_id=trial_id,
                              split="train",
                              features=selected_features,
                              regime=min_regime,
                              ret_name=trade_params["ret_name"],
                              expected_code="HC")

    name = _run_identity(trade_params, env_params, model_params, train_params,
                         selected_features, min_regime, daily_regime)

    output_dir = os.path.join(base_path, method, instruments, "temp", "model",
                              str(task_id), str(period),
                              "rl", "hybrid_transformer_loss", "result",
                              str(name), "feature_correlation_audit")

    print(
        f"[FEATURE_CORR_START] run_id={name} RB={len(left_train)} HC={len(right_train)} "
        f"old={len(old_features)} new={len(new_features)}")

    return feature_correlation({
        "RB": left_train,
        "HC": right_train
    },
                               old_features,
                               new_features,
                               output_dir,
                               strong_threshold=float(strong_threshold),
                               duplicate_threshold=float(duplicate_threshold))


if __name__ == "__main__":
    variant = Tactix().start()
    run(method=variant.method,
        instruments=variant.instruments,
        task_id=variant.task_id,
        period=variant.period,
        env_id=variant.env_id,
        trade_id=variant.trade_id,
        model_id=variant.model_id,
        train_id=variant.train_id,
        feature_id=variant.feature_id,
        trial_id='10002',
        regime_id=variant.regime_id)
