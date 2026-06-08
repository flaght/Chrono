"""
特征对比测试脚本
目的：在同一测试集上比较原始267维特征 vs AE 64维输出
包含：RankIC、方向准确率、自相关性分析、有效样本数估算

运行方式：与 3.2.1.autoencoder.py 相同的参数
"""
import copy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from dotenv import load_dotenv

load_dotenv()

from lib.HybridTransformer.transformer import TemporientTransformer
from lib.uvx import *
from lib.syn005.trainer import Trainer
from lib.syn005.evaluator import Evaluator
from kdutils.macro2 import *
from kdutils.tactix import Tactix


def compute_autocorrelation(series: np.ndarray, max_lag: int = 15) -> dict:
    """
    计算时间序列的自相关性

    Args:
        series: 1D 数组
        max_lag: 最大滞后期数

    Returns:
        自相关统计
    """
    autocorrs = []
    for lag in range(1, max_lag + 1):
        if len(series) > lag:
            corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            if not np.isnan(corr):
                autocorrs.append(corr)
            else:
                autocorrs.append(0.0)
        else:
            autocorrs.append(0.0)

    lag1_autocorr = autocorrs[0] if autocorrs else 0

    return {
        'lag1': lag1_autocorr,
        'lag5': autocorrs[4] if len(autocorrs) > 4 else 0,
        'lag15': autocorrs[14] if len(autocorrs) > 14 else 0,
        'decay_curve': autocorrs
    }


def compute_effective_samples(n_samples: int, autocorr: float) -> int:
    """
    基于自相关性计算有效样本数

    公式: N_eff = N * (1 - rho) / (1 + rho)

    Args:
        n_samples: 原始样本数
        autocorr: lag-1 自相关系数

    Returns:
        有效样本数
    """
    if autocorr >= 1.0:
        return 1
    if autocorr <= -1.0:
        return n_samples

    rho = abs(autocorr)
    n_eff = n_samples * (1 - rho) / (1 + rho)
    return max(1, int(n_eff))


def rank_ic_analysis(features: np.ndarray, target: np.ndarray) -> dict:
    """
    RankIC 分析

    Args:
        features: 特征矩阵 (N, D)
        target: 目标变量 (N,)

    Returns:
        RankIC 统计
    """
    n_features = features.shape[1]
    rank_ics = []

    for i in range(n_features):
        feat = features[:, i]
        if np.std(feat) < 1e-8:
            rank_ics.append(0.0)
            continue

        ric, _ = spearmanr(feat, target)
        if not np.isnan(ric):
            rank_ics.append(ric)
        else:
            rank_ics.append(0.0)

    rank_ics = np.array(rank_ics)
    abs_rank_ics = np.abs(rank_ics)

    max_ic = np.max(abs_rank_ics)
    mean_ic = np.mean(abs_rank_ics)
    best_idx = np.argmax(abs_rank_ics)

    return {
        'max': float(max_ic),
        'mean': float(mean_ic),
        'median': float(np.median(abs_rank_ics)),
        'best_feature_idx': int(best_idx),
        'best_feature_ic': float(rank_ics[best_idx]),
        'n_significant_01': int(np.sum(abs_rank_ics > 0.01)),
        'n_significant_02': int(np.sum(abs_rank_ics > 0.02)),
        'n_significant_03': int(np.sum(abs_rank_ics > 0.03)),
    }


def direction_accuracy(features: np.ndarray, target: np.ndarray, best_idx: int) -> dict:
    """
    预测方向准确率
    """
    best_feat = features[:, best_idx]
    sign_match = np.mean(np.sign(best_feat) == np.sign(target))

    return {
        'direction_accuracy': float(sign_match),
    }


def run_comparison(method, task_id, instruments, period, name,
                   nan_threshold, var_threshold, corr_threshold, ic_threshold):
    """
    主测试函数
    """
    print("=" * 80)
    print("特征对比测试：原始 267 维 vs AE 64 维")
    print("=" * 80)

    FEATURE_PARAMS = {
        'nan_threshold': nan_threshold,
        'var_threshold': var_threshold,
        'corr_threshold': corr_threshold,
        'ic_threshold': ic_threshold
    }

    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                           str(task_id), str(period), "research")

    # 获取特征列表
    features_df = fetch_research_fetures(
        method=method, instruments=instruments, task_id=task_id,
        period=period, name='feature', params=FEATURE_PARAMS)
    selected_features = features_df['feature'].tolist()

    # 获取测试数据
    _, test_data = fetch_clean_data2(
        method=method, task_id=task_id, instruments=instruments,
        output=outdirs, params={'nan_threshold': nan_threshold, 'var_threshold': var_threshold})

    feature_dim = len(selected_features)
    print(f"\n原始特征数: {feature_dim}")

    # 加载参数
    AUTOENCODE_PARAMS, TRAIN_PARAMS = load_params(
        file_dirs=outdirs, name="autoencode", model_name='params1', train_name="params1")
    AUTOENCODE_PARAMS['enc_in'] = feature_dim

    # 准备数据
    trainer = Trainer(params=AUTOENCODE_PARAMS, train_params=TRAIN_PARAMS,
                      output_dirs=outdirs, name=name)

    X_raw, y, dates = trainer.prepare_data(test_data, selected_features,
                                            "nxt1_ret_{}h".format(period))

    print(f"\n测试集样本数: {len(y)}")
    print(f"时间范围: {dates[0]} ~ {dates[-1]}")

    # ========== 1. Y 的自相关性分析 ==========
    print("\n" + "=" * 80)
    print("1. 目标变量 Y 的自相关性分析")
    print("=" * 80)

    y_autocorr = compute_autocorrelation(y, max_lag=15)
    n_eff_y = compute_effective_samples(len(y), y_autocorr['lag1'])

    print(f"  Y 的 Lag-1 自相关: {y_autocorr['lag1']:.4f}")
    print(f"  Y 的 Lag-5 自相关: {y_autocorr['lag5']:.4f}")
    print(f"  Y 的 Lag-15 自相关: {y_autocorr['lag15']:.4f}")
    print(f"  原始样本数: {len(y):,}")
    print(f"  有效样本数 (基于 Lag-1): {n_eff_y:,}")
    print(f"  有效率: {n_eff_y / len(y) * 100:.1f}%")

    # ========== 2. 原始特征分析 ==========
    print("\n" + "=" * 80)
    print("2. 原始 267 维特征分析 (测试集)")
    print("=" * 80)

    # 原始特征的 RankIC
    raw_rank_ic = rank_ic_analysis(X_raw, y)
    raw_dir_acc = direction_accuracy(X_raw, y, raw_rank_ic['best_feature_idx'])

    print(f"  Max RankIC: {raw_rank_ic['max']:.4f}")
    print(f"  Mean RankIC: {raw_rank_ic['mean']:.4f}")
    print(f"  最佳特征索引: {raw_rank_ic['best_feature_idx']}")
    print(f"  最佳特征 IC: {raw_rank_ic['best_feature_ic']:.4f}")
    print(f"  IC > 0.01 的特征数: {raw_rank_ic['n_significant_01']}")
    print(f"  IC > 0.02 的特征数: {raw_rank_ic['n_significant_02']}")
    print(f"  IC > 0.03 的特征数: {raw_rank_ic['n_significant_03']}")
    print(f"  方向准确率: {raw_dir_acc['direction_accuracy']:.1%}")

    # 原始特征的自相关性 (最佳特征)
    best_raw_feat = X_raw[:, raw_rank_ic['best_feature_idx']]
    raw_feat_autocorr = compute_autocorrelation(best_raw_feat)
    print(f"  最佳特征 Lag-1 自相关: {raw_feat_autocorr['lag1']:.4f}")

    # ========== 3. AE 特征分析 ==========
    print("\n" + "=" * 80)
    print("3. AE 64 维特征分析 (测试集)")
    print("=" * 80)

    # 创建滚动窗口样本
    test_samples = trainer.create_rolling_window_samples(X_raw)
    test_loader = trainer.create_predict_data_loader(test_samples)

    # 生成 AE 隐层特征
    factors_array, original_array, reconstructed_array = trainer.predict(
        model_method=TemporientTransformer,
        data_loader=test_loader,
        multi_timestep_extraction=False
    )

    # 对齐时间戳
    seq_len = TRAIN_PARAMS['seq_len']
    aligned_y = y[seq_len - 1:]

    print(f"  AE 输出形状: {factors_array.shape}")
    print(f"  对齐后样本数: {len(aligned_y)}")

    # AE 特征的 RankIC
    ae_rank_ic = rank_ic_analysis(factors_array, aligned_y)
    ae_dir_acc = direction_accuracy(factors_array, aligned_y, ae_rank_ic['best_feature_idx'])

    print(f"  Max RankIC: {ae_rank_ic['max']:.4f}")
    print(f"  Mean RankIC: {ae_rank_ic['mean']:.4f}")
    print(f"  最佳特征索引: {ae_rank_ic['best_feature_idx']}")
    print(f"  最佳特征 IC: {ae_rank_ic['best_feature_ic']:.4f}")
    print(f"  IC > 0.01 的特征数: {ae_rank_ic['n_significant_01']}")
    print(f"  IC > 0.02 的特征数: {ae_rank_ic['n_significant_02']}")
    print(f"  IC > 0.03 的特征数: {ae_rank_ic['n_significant_03']}")
    print(f"  方向准确率: {ae_dir_acc['direction_accuracy']:.1%}")

    # AE 特征的自相关性
    best_ae_feat = factors_array[:, ae_rank_ic['best_feature_idx']]
    ae_feat_autocorr = compute_autocorrelation(best_ae_feat)
    print(f"  最佳特征 Lag-1 自相关: {ae_feat_autocorr['lag1']:.4f}")

    # ========== 4. 对比总结 ==========
    print("\n" + "=" * 80)
    print("4. 对比总结")
    print("=" * 80)

    print("\n  | 指标 | 原始 267 维 | AE 64 维 | 变化 |")
    print("  |------|------------|----------|------|")
    print(f"  | Max RankIC | {raw_rank_ic['max']:.4f} | {ae_rank_ic['max']:.4f} | {(ae_rank_ic['max'] - raw_rank_ic['max']) / raw_rank_ic['max'] * 100:+.1f}% |")
    print(f"  | Mean RankIC | {raw_rank_ic['mean']:.4f} | {ae_rank_ic['mean']:.4f} | {(ae_rank_ic['mean'] - raw_rank_ic['mean']) / raw_rank_ic['mean'] * 100:+.1f}% |")
    print(f"  | 方向准确率 | {raw_dir_acc['direction_accuracy']:.1%} | {ae_dir_acc['direction_accuracy']:.1%} | {(ae_dir_acc['direction_accuracy'] - raw_dir_acc['direction_accuracy']) * 100:+.1f}pp |")
    print(f"  | IC>0.01 数 | {raw_rank_ic['n_significant_01']} | {ae_rank_ic['n_significant_01']} | - |")
    print(f"  | IC>0.03 数 | {raw_rank_ic['n_significant_03']} | {ae_rank_ic['n_significant_03']} | - |")
    print(f"  | Lag-1 自相关 | {raw_feat_autocorr['lag1']:.4f} | {ae_feat_autocorr['lag1']:.4f} | - |")

    # ========== 5. 有效样本数与模型容量分析 ==========
    print("\n" + "=" * 80)
    print("5. 有效样本数与模型容量分析")
    print("=" * 80)

    # 基于持仓时间估算
    holding_period = int(period)  # 15 分钟
    n_raw = len(y)
    n_eff_holding = n_raw // holding_period

    print(f"\n  持仓周期: {holding_period} 分钟")
    print(f"  原始样本数: {n_raw:,}")
    print(f"  基于持仓周期的有效样本数: {n_eff_holding:,}")
    print(f"  基于 Y 自相关的有效样本数: {n_eff_y:,}")

    # 模型容量建议
    # 经验法则: 每层需要约 10K-50K 有效样本
    max_layers_conservative = n_eff_y // 50000
    max_layers_aggressive = n_eff_y // 10000

    print(f"\n  模型层数建议:")
    print(f"    保守估计 (50K/层): {max_layers_conservative} 层")
    print(f"    激进估计 (10K/层): {max_layers_aggressive} 层")
    print(f"    当前使用: 4 层")

    # ========== 6. 结论 ==========
    print("\n" + "=" * 80)
    print("6. 诊断结论")
    print("=" * 80)

    if raw_rank_ic['max'] < 0.02:
        print("\n  ⚠️ 原始特征预测能力弱 (Max IC < 2%)")
    if ae_rank_ic['max'] < raw_rank_ic['max'] * 0.8:
        print("  ⚠️ AE 压缩损失超过 20% 信号")
    if raw_dir_acc['direction_accuracy'] < 0.51:
        print("  ⚠️ 原始特征方向准确率 < 51% (接近随机)")
    if ae_dir_acc['direction_accuracy'] < 0.50:
        print("  ⚠️ AE 特征方向准确率 < 50% (比随机还差)")
    if y_autocorr['lag1'] > 0.5:
        print(f"  ⚠️ Y 高自相关 ({y_autocorr['lag1']:.2f})，有效样本数大幅减少")

    return {
        'raw': raw_rank_ic,
        'ae': ae_rank_ic,
        'y_autocorr': y_autocorr,
        'n_effective': n_eff_y
    }


if __name__ == '__main__':
    variant = Tactix().start()

    results = run_comparison(
        method=variant.method,
        instruments=variant.instruments,
        task_id=variant.task_id,
        period=variant.period,
        name=variant.name,
        nan_threshold=0.5,
        var_threshold=1e-10,
        corr_threshold=0.95,
        ic_threshold=0.01
    )
