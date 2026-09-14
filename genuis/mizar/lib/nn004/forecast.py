import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kdutils.macro2 import *

from .inference import (load_checkpoint, model_config_from_checkpoint,
                        predict_store, resolve_device)
from .data import SequenceStore
from .model import HybridTransformerRegressor
from .metrics import evaluate_predictions


def _result_dir(method, instruments, task_id, period, name):
    return os.path.join(base_path, method, instruments, "temp", "model",
                        str(task_id), str(period), "rl",
                        "hybrid_transformer_loss", "result", str(name))


def find_model_files(method, instruments, task_id, period, model_id):
    """由主task_id和训练结果ID定位模型与固化参数文件。"""
    task_key, model_id = str(task_id), str(model_id)
    if not model_id.isdigit():
        raise ValueError("run_id必须是数字训练结果ID")
    run_dir = _result_dir(method, instruments, task_key, period, model_id)
    files = {
        "run_dir": run_dir,
        "model_path": os.path.join(run_dir, "models", "best_model.pt"),
        "config_path": os.path.join(run_dir, "config.json"),
    }
    missing = [
        path for key, path in files.items()
        if key != "run_dir" and not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError("训练结果文件不存在: " + ", ".join(missing))
    return files


def equal_weight_ensemble(predictions):
    """对任意多个已对齐模型的 ``predicted_z`` 做等权平均。"""
    if len(predictions) < 2:
        raise ValueError('等权融合至少需要两个模型')
    model_keys = list(predictions)
    frames = []
    for model_key in model_keys:
        frame = predictions[model_key].copy()
        frame['trade_time'] = pd.to_datetime(frame['trade_time'],
                                             errors='raise')
        if frame[['code', 'trade_time']].isna().any().any():
            raise ValueError('预测键不能缺失')
        if frame.duplicated(['code', 'trade_time']).any():
            raise ValueError('预测键重复')
        frame = frame.sort_values(['code',
                                   'trade_time']).reset_index(drop=True)
        if not np.isfinite(frame['predicted_z']).all():
            raise ValueError(f'模型{model_key}存在无效预测，不能静默过滤')
        frames.append(frame)
    base = frames[0]
    for frame in frames[1:]:
        if not base[['code', 'trade_time', 'evaluation_segment']].equals(
                frame[['code', 'trade_time', 'evaluation_segment']]):
            raise ValueError('各模型的预测样本或评价分段不一致')
        for column in ('future_ret_h', 'ret_scale', 'prediction_bound_std'):
            if not np.allclose(
                    base[column], frame[column], rtol=0, atol=0,
                    equal_nan=True):
                raise ValueError(f'各模型的{column}不一致')
    result = base.copy()
    result['predicted_z'] = np.mean(
        [frame['predicted_z'].to_numpy() for frame in frames], axis=0)
    result['predicted_ret_h'] = result.predicted_z * result.ret_scale
    result['target_z'] = result.future_ret_h / result.ret_scale
    result['prediction_error_z'] = result.predicted_z - result.target_z
    result['squared_error_z'] = result.prediction_error_z**2
    result['action_raw'] = result.predicted_z / result.prediction_bound_std
    result['direction'] = np.sign(result.predicted_ret_h).astype(np.int8)
    result['confidence'] = result.predicted_z.abs()
    result['er_value'] = result['net_er_out'] = result.predicted_z
    return result


def compare_seeds(run_dirs,
                  asset_paths,
                  output_dir,
                  split='val',
                  batch_size=4096,
                  device='auto',
                  ret_name='nxt1_ret_5h',
                  return_predictions=False):
    if split not in ('val', 'test') or batch_size <= 0:
        raise ValueError('split必须为val/test，batch_size必须大于0')
    if set(asset_paths) != {'RB', 'HC'}:
        raise ValueError('本入口必须同时评价RB、HC')
    if len(run_dirs) < 2:
        raise ValueError('至少需要指定两个seed及其训练目录')
    seeds = list(run_dirs)
    configs = {}
    for seed in seeds:
        with open(Path(run_dirs[seed]) / 'config.json', encoding='utf-8') as f:
            config = json.load(f)
        if config['env_config']['seed'] != seed:
            raise ValueError(f'seed={seed}目录中的实际seed不匹配')
        loss = config['loss_config']
        if (loss['loss_type'] != 'huber_corr' or loss['huber_delta'] != 1.0
                or loss['corr_weight'] != 0):
            raise ValueError('本入口要求纯Huber(delta=1,corr_weight=0)')
        configs[seed] = config
    reference = configs[seeds[0]]
    for config in configs.values():
        for field in ('features', 'lookback', 'ret_scale_by_code',
                      'architecture', 'model_config', 'train_config'):
            if config[field] != reference[field]:
                raise ValueError(f'配置字段{field}不一致，不能作为同配置三seed比较')
    parts = []
    for code, path in asset_paths.items():
        required_columns = [
            'trade_time', 'code', ret_name, *reference['features']
        ]
        try:
            frame = pd.read_feather(path, columns=required_columns)
        except (KeyError, ValueError) as error:
            raise ValueError(f'{path}缺少推理所需字段: {error}') from error
        if set(frame.code.astype(str)) != {code}:
            raise ValueError(f'{path}必须仅包含{code}')
        if ret_name != 'nxt1_ret' and 'nxt1_ret' in frame:
            raise ValueError('存在nxt1_ret与指定标签同名冲突')
        frame = frame.rename(columns={ret_name: 'nxt1_ret'})
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    store = SequenceStore(data,
                          reference['features'],
                          reference['lookback'],
                          ret_scale_by_code=reference['ret_scale_by_code'])
    runtime = resolve_device(device)
    validation_device = resolve_device('cpu')
    checkpoints = {}
    architectures = {}
    # 先在CPU完成所有模型的完整性和兼容性检查。任何模型有问题时，
    # 不创建输出目录，修正后可以使用同一output_dir直接重试。
    for seed in seeds:
        path = Path(run_dirs[seed]) / 'models' / 'best_model.pt'
        checkpoint = load_checkpoint(str(path), validation_device)
        if checkpoint.get('version') != 'hybrid_transformer_loss_v1':
            raise ValueError(f'{path}版本错误')
        for field in ('features', 'lookback', 'ret_scale_by_code',
                      'loss_config'):
            if checkpoint[field] != configs[seed][field]:
                raise ValueError(f'{path}与config.json的{field}不一致')
        architecture = model_config_from_checkpoint(checkpoint)
        if architecture != configs[seed]['architecture']:
            raise ValueError(f'{path}网络结构与config.json不一致')
        checkpoints[seed] = checkpoint
        architectures[seed] = architecture
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    predictions = {}
    for seed in seeds:
        checkpoint = checkpoints.pop(seed)
        architecture = architectures.pop(seed)
        model = HybridTransformerRegressor(**architecture).to(runtime)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(
            f'[COMPARE] split={split} seed={seed} rows={len(data)} '
            f'batch_size={batch_size} device={runtime}',
            flush=True)
        predictions[seed] = predict_store(model, store, batch_size, runtime)
        del model, checkpoint
    schemes = {f'seed_{seed}': predictions[seed] for seed in seeds}
    schemes['ensemble_equal'] = equal_weight_ensemble(predictions)
    settings = reference['train_config']
    comparison, periods = [], []
    for name, records in schemes.items():
        records.to_csv(destination / f'{name}_predictions.csv', index=False)
        metrics = evaluate_predictions(records,
                                       settings['ic_resampling_minutes'],
                                       settings['ic_roll_window'],
                                       settings['ic_min_periods'])
        sequence = metrics.pop('ic_sequence')
        sequence.to_csv(destination / f'{name}_rolling_ic.csv', index=False)
        with open(destination / f'{name}_metrics.json', 'w',
                  encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        for code, values in metrics['assets'].items():
            comparison.append(
                dict(scheme=name,
                     code=code,
                     selection_score=metrics['selection_score'],
                     **values))
        for frequency in ('M', 'Q'):
            part = sequence.copy()
            part['period'] = pd.to_datetime(
                part.trade_time).dt.to_period(frequency).astype(str)
            for (code, period), group in part.groupby(['code', 'period']):
                values = group.rolling_pearson_ic.dropna()
                periods.append(
                    dict(scheme=name,
                         code=code,
                         frequency=frequency,
                         period=period,
                         rolling_ic_mean=values.mean(),
                         valid_windows=len(values)))
    pd.DataFrame(comparison).to_csv(destination / 'comparison.csv',
                                    index=False)
    pd.DataFrame(periods).to_csv(destination / 'period_comparison.csv',
                                 index=False)
    manifest = dict(
        split=split,
        run_dirs={
            str(k): str(v)
            for k, v in run_dirs.items()
        },
        asset_paths={
            k: str(v)
            for k, v in asset_paths.items()
        },
        rows=len(data),
        inference_batch_size=batch_size,
        ensemble_weights={str(seed): 1 / len(seeds)
                          for seed in seeds},
        label=ret_name,
        selection='report_only_no_automatic_winner')
    with open(destination / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(
        pd.DataFrame(comparison)[[
            'scheme', 'code', 'selection_score', 'rolling_pearson_ic_mean',
            'total_pearson_ic'
        ]])
    if return_predictions:
        return dict(predictions=schemes,
                    comparison=pd.DataFrame(comparison),
                    periods=pd.DataFrame(periods),
                    output_dir=str(destination))
    return pd.DataFrame(comparison)
