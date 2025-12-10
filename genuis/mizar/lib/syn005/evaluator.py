from typing import Dict,List
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kurtosis
from sklearn.feature_selection import mutual_info_regression
from lib.svx001 import scale_factors
from lib import logger

class Evaluator(object):
    def __init__(self):
        pass


    def _standardize(self, data: np.ndarray, win: int) -> np.ndarray:
        df = pd.DataFrame(data)
        # scale_factors requires column names
        df.columns = [f'feat_{i}' for i in range(data.shape[1])]
        std_data = np.zeros_like(data)
        for i, col in enumerate(df.columns):
            # scale_factors modifies df in-place, adding 'transformed'
            # We use 'roll_zscore' as per design
            scale_factors(df, 'roll_zscore', win, col)
            std_data[:, i] = df['transformed'].values
            
        return std_data

    def rank_ic(self, latent_features: np.ndarray, 
                        target: np.ndarray) -> Dict:
        """
        RankIC 分析
        
        Args:
            latent_features: 潜在特征 (N, D)
            target: 目标收益率 (N,)
        
        | 字段 | 含义 | 作用 | 示例值 |
        |------|------|------|--------|
        | **max** | 所有 latent 维度中，与 target 相关性最高的 IC 值 | **核心判断指标**，决定 latent 是否有预测价值 | 0.0234 |
        | **mean** | 所有 latent 维度 IC 的平均值 | 衡量整体质量，mean 高说明多个维度都有信息 | 0.0089 |
        | **best_feature_idx** | IC 最高的那个维度的索引 | 用于后续分析（自相关、方向等都基于这个维度） | 42 |
        | **quality** | IC 质量等级 | 快速判断，便于报告 | 'good' |
        | **n_significant** | IC > 0.01 的维度数量 | 衡量**有效特征数**，多个显著维度 = 信息更丰富 | 156 |
        """
        
        n_features = latent_features.shape[1]
        rank_ics = []
        
        for i in range(n_features):
            feat = latent_features[:, i]
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
            'n_significant_01': int(np.sum(abs_rank_ics > 0.01)),
            'n_significant_02': int(np.sum(abs_rank_ics > 0.02)),
            'rank_ics': rank_ics.tolist()
        }
    
    def direction_accuracy(self, latent_features: np.ndarray,
                                    target: np.ndarray,
                                    best_idx: int) -> Dict:
        """
        预测方向准确率
        
        Args:
            latent_features: 潜在特征 (N, D)
            target: 目标收益率 (N,)
            best_idx: 最佳特征索引
        
        Returns:
            方向准确率统计
        """
        best_feat = latent_features[:, best_idx]
        # 方法1: 符号匹配
        sign_match = np.mean(np.sign(best_feat) == np.sign(target))
        
        # 方法2: 分组验证(更稳健)
        n_groups = 10
        sorted_idx = np.argsort(best_feat)
        group_size = len(target) // n_groups

        correct_groups = 0
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else len(target)
            group_return = np.mean(target[sorted_idx[start:end]])
            expected_sign = 1 if i >= n_groups // 2 else -1
            if np.sign(group_return) == expected_sign or abs(group_return) < 1e-8:
                correct_groups += 1

        group_direction_acc = correct_groups / n_groups
        # 修复BUG: 不再取max，直接返回真实的符号匹配
        # 分组验证衡量的是排序能力，而非方向准
        #direction_acc = max(sign_match, group_direction_acc)
        direction_acc = sign_match

        return {
            'direction_accuracy': float(direction_acc),  # 真实的方向预测准确率
            'sign_match': float(sign_match),             # 符号匹配（与direction_accuracy相同）
            'ranking_ability': float(group_direction_acc) # 排序能力（原group_accuracy）
        }


    def ic_decay(self, feature: np.ndarray, 
                         target: np.ndarray,
                         ic_decay_lags:int = 10) -> Dict:
        """
        IC 衰减分析
        
        Args:
            feature: 单个特征 (N,)
            target: 目标收益率 (N,)
        
        Returns:
            IC 衰减统计
        """
        decay_curve = []
        
        for lag in range(ic_decay_lags):
            if lag == 0:
                ic = spearmanr(feature, target)[0]
            else:
                if len(feature) > lag:
                    ic = spearmanr(feature[:-lag], target[lag:])[0]
                else:
                    ic = 0
            
            decay_curve.append(abs(ic) if not np.isnan(ic) else 0)
        
        # 计算半衰期
        if decay_curve[0] > 0:
            half_life = next(
                (i for i, v in enumerate(decay_curve) if v < decay_curve[0] * 0.5),
                ic_decay_lags
            )
        else:
            half_life = 0
        
        return {
            'value': int(half_life),
            'decay_curve': decay_curve,
            'decay_rate': float((decay_curve[0] - decay_curve[-1]) / (decay_curve[0] + 1e-8)),

        }
    
    def tail_prediction(self, feature: np.ndarray,
                                 target: np.ndarray, tail_percentile:int=90) -> Dict:
        """
        尾部预测分析(极端行情表现)
        """
        volatility = np.abs(target)
        threshold = np.percentile(volatility, tail_percentile)
        
        high_vol_mask = volatility > threshold
        normal_mask = ~high_vol_mask
        
        overall_ic = abs(spearmanr(feature, target)[0])
        
        if np.sum(high_vol_mask) > 10:
            high_vol_ic = abs(spearmanr(feature[high_vol_mask], target[high_vol_mask])[0])
        else:
            high_vol_ic = overall_ic
        
        if np.sum(normal_mask) > 10:
            normal_ic = abs(spearmanr(feature[normal_mask], target[normal_mask])[0])
        else:
            normal_ic = overall_ic
        
        ic_drop = (normal_ic - high_vol_ic) / (normal_ic + 1e-8)
        is_crisis_alpha = high_vol_ic > normal_ic * 1.1
        
        return {
            'overall_ic': float(overall_ic),
            'high_vol_ic': float(high_vol_ic),
            'normal_ic': float(normal_ic),
            'ic_drop': float(ic_drop),
            'is_crisis_alpha': is_crisis_alpha,
            'high_vol_samples': int(np.sum(high_vol_mask))
        }
        

    def dead_neurons(self, latent: np.ndarray, threshold: float = 1e-6) -> Dict:
        """
        检测死神经元
        
        Args:
            latent: 潜在表示 (N, latent_dim)
            threshold: 方差阈值
        
        Returns:
            死神经元统计
        """
        variances = np.var(latent, axis=0)
        dead_indices = np.where(variances < threshold)[0]
        dead_ratio = len(dead_indices) / latent.shape[1]
        
        return {
            "dead_neuron_count": len(dead_indices),
            "dead_neuron_ratio": dead_ratio,
            "dead_indices": dead_indices.tolist(),
        }
    
    def autocorrelation(self, latent_features: np.ndarray,
                                best_idx: int) -> Dict:
        """
        自相关分析
        
        Args:
            latent_features: 潜在特征 (N, D)
            best_idx: 最佳特征索引
        
        Returns:
            自相关统计
        """
        variances = np.var(latent_features, axis=0)
        top_var_idx = np.argsort(variances)[-5:]
        check_indices = list(set([best_idx] + top_var_idx.tolist()))
        
        autocorrs = []
        for idx in check_indices:
            if variances[idx] > 1e-6:
                ac = np.corrcoef(latent_features[:-1, idx], latent_features[1:, idx])[0, 1]
                if not np.isnan(ac):
                    autocorrs.append(ac)
        
        max_autocorr = max(autocorrs) if autocorrs else 0
        mean_autocorr = np.mean(autocorrs) if autocorrs else 0
        
        return {
            'max': float(max_autocorr),
            'mean': float(mean_autocorr)
        }
    
    def reconstruction(self, original: np.ndarray, 
                      reconstructed: np.ndarray) -> Dict:
        """
        重建质量评估
        
        包含:
        1. 基础指标(MSE, MAE, EV)
        2. 方向准确率(沿seq_len维度)
        3. 最后时间步质量
        4. 分位数误差
        
        Args:
            original: 原始输入 (N, seq_len, num_features) 或 (N, num_features)
            reconstructed: 重建输出,形状同 original
        
        Returns:
            重建质量指标
        """
        results = {}
        is_3d = original.ndim == 3
        
        # ==================== 1. 基础重建指标 ====================
        errors = original - reconstructed
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        
        # Explained Variance
        var_original = np.var(original)
        var_error = np.var(errors)
        explained_variance = 1 - (var_error / (var_original + 1e-8))
        
        if is_3d:
            N, seq_len, num_features = original.shape
            
            # 按时间步的误差分布(检查是否序列末端重建更差)
            mse_per_timestep = np.mean(errors ** 2, axis=(0, 2))
            
            # 按时间步的 EV
            ev_per_step = []
            for t in range(seq_len):
                err_t = original[:, t, :] - reconstructed[:, t, :]
                ev_t = 1 - np.var(err_t) / (np.var(original[:, t, :]) + 1e-8)
                ev_per_step.append(ev_t)
            
            ev_first_5 = np.mean(ev_per_step[:5])
            ev_last_5 = np.mean(ev_per_step[-5:])
            ev_trend = 'degrading' if ev_last_5 < ev_first_5 * 0.9 else 'stable'
            
            results['basic'] = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse)),
                'explained_variance': float(explained_variance),
                'mse_first_step': float(mse_per_timestep[0]),
                'mse_last_step': float(mse_per_timestep[-1]),
                'mse_trend': 'degrading' if mse_per_timestep[-1] > mse_per_timestep[0] * 1.2 else 'stable',
                'ev_first_5_steps': float(ev_first_5),
                'ev_last_5_steps': float(ev_last_5),
                'ev_trend': ev_trend
            }
        else:
            results['basic'] = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse)),
                'explained_variance': float(explained_variance)
            }
        
        # ==================== 2. 方向准确率(沿seq_len维度) ====================
        if is_3d:
            # 计算每个样本内部的时序方向
            # diff along seq_len: (N, seq_len-1, num_features)
            true_diff = np.diff(original, axis=1)
            pred_diff = np.diff(reconstructed, axis=1)
            
            true_direction = np.sign(true_diff)
            pred_direction = np.sign(pred_diff)
            
            # 整体方向准确率
            direction_match = (true_direction == pred_direction)
            overall_da = np.mean(direction_match)
            da_per_timestep = np.mean(direction_match, axis=(0, 2))
            da_per_feature = np.mean(direction_match, axis=(0, 1))
            
            # 加权方向准确率(大波动更重要)
            weights = np.abs(true_diff)
            weights_normalized = weights / (np.sum(weights) + 1e-8)
            weighted_da = np.sum(weights_normalized * direction_match)
            
            results['directional'] = {
                'overall_accuracy': float(overall_da),
                'weighted_accuracy': float(weighted_da),
                'da_first_steps': float(np.mean(da_per_timestep[:5])),   # 前5步
                'da_last_steps': float(np.mean(da_per_timestep[-5:])),   # 后5步
                'da_best_feature': float(np.max(da_per_feature)),
                'da_worst_feature': float(np.min(da_per_feature)),
                'n_good_features': int(np.sum(da_per_feature > 0.55))   # DA > 55% 的特征数
            }
        else:
            # 2D 数据
            true_diff = np.diff(original, axis=0)
            pred_diff = np.diff(reconstructed, axis=0)
            true_dir = np.sign(true_diff)
            pred_dir = np.sign(pred_diff)
            
            overall_da = np.mean(true_dir == pred_dir)
            
            results['directional'] = {
                'overall_accuracy': float(overall_da),
                'weighted_accuracy': float(overall_da)
            }
        
        # ==================== 3. 最后时间步质量(关键!) ====================
        if is_3d:
            orig_last = original[:, -1, :]
            recon_last = reconstructed[:, -1, :]
        else:
            orig_last = original
            recon_last = reconstructed
        
        err_last = orig_last - recon_last
        ev_last = 1 - np.var(err_last) / (np.var(orig_last) + 1e-8)
        mse_last = np.mean(err_last ** 2)
        mae_last = np.mean(np.abs(err_last))
        
        # 相关结构保留
        try:
            corr_orig = np.corrcoef(orig_last.T)
            corr_recon = np.corrcoef(recon_last.T)
            n = corr_orig.shape[0]
            triu_idx = np.triu_indices(n, k=1)
            corr_preservation = np.corrcoef(
                corr_orig[triu_idx], corr_recon[triu_idx])[0, 1]
        except:
            corr_preservation = 0.0
        
        # P99 误差
        p99_error = np.percentile(np.abs(err_last), 99)
        
        results['last_step'] = {
            'ev': float(ev_last),
            'mse': float(mse_last),
            'mae': float(mae_last),
            'correlation_preserved': float(corr_preservation),
            'p99_error': float(p99_error)
        }
        
        # ==================== 4. 分位数误差 ====================
        abs_errors = np.abs(errors).flatten()
        
        results['quantile_errors'] = {
            'median': float(np.median(abs_errors)),
            'p90': float(np.percentile(abs_errors, 90)),
            'p99': float(np.percentile(abs_errors, 99)),
            'max': float(np.max(abs_errors))
        }
    
        
        return results
    
    def effective_dimensions(self, latent_features: np.ndarray,
                                      dead_indices: List[int]) -> Dict:
        """
        有效维度分析
        
        Args:
            latent_features: 潜在特征 (N, D)
            dead_indices: 死神经元索引列表
        
        Returns:
            有效维度统计
        """
        dead_mask = np.zeros(latent_features.shape[1], dtype=bool)
        dead_mask[dead_indices] = True
        
        valid_latent = latent_features[:, ~dead_mask]
        
        if valid_latent.shape[1] == 0:
            return {'error': 'All neurons are dead'}
        
        try:
            centered = valid_latent - valid_latent.mean(axis=0)
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            var_explained = s ** 2 / np.sum(s ** 2)
            cumsum = np.cumsum(var_explained)
            dims_95 = np.searchsorted(cumsum, 0.95) + 1
            utilization = dims_95 / valid_latent.shape[1]
            
            return {
                'dims_95': int(dims_95),
                'total': int(valid_latent.shape[1]),
                'utilization': float(utilization),
                'top_5_variance': var_explained[:5].tolist()
            }
        except Exception as e:
            return {'error': str(e)}
        
    def mutual_information(self, latent_features: np.ndarray,
                                    target: np.ndarray,
                                    original: np.ndarray = None) -> Dict:
        """
        互信息分析
        
        评估潜在表示与目标的互信息,以及与原始特征的对比
        
        Args:
            latent_features: 潜在特征 (N, D)
            target: 目标收益率 (N,)
            original: 原始输入 (N, seq_len, features) 或 (N, features)
        
        Returns:
            互信息统计
        """
        
        results = {}
        
        try:
            # 潜在特征与目标的互信息
            mi_latent = mutual_info_regression(latent_features, target, 
                                               n_neighbors=5, random_state=42)
            
            results['latent_mi'] = {
                'total_mi': float(np.sum(mi_latent)),
                'mean_mi': float(np.mean(mi_latent)),
                'max_mi': float(np.max(mi_latent)),
                'top_mi_indices': np.argsort(mi_latent)[-5:][::-1].tolist()
            }
            
            # 与原始特征对比
            if original is not None:
                # 如果是3D,取最后时间步
                if original.ndim == 3:
                    original_2d = original[:, -1, :]
                else:
                    original_2d = original
                
                mi_original = mutual_info_regression(original_2d, target,
                                                    n_neighbors=5, random_state=42)
                
                results['original_mi'] = {
                    'total_mi': float(np.sum(mi_original)),
                    'mean_mi': float(np.mean(mi_original))
                }
                
                # 信息效率(压缩后的信息保留)
                compression_ratio = original_2d.shape[1] / latent_features.shape[1]
                info_efficiency = np.mean(mi_latent) / (np.mean(mi_original) + 1e-8)
                
                results['comparison'] = {
                    'compression_ratio': float(compression_ratio),
                    'information_efficiency': float(info_efficiency),
                    'info_preserved': info_efficiency >= 0.8
                }
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def final_metrics(self, latent_features: np.ndarray, 
                     target: np.ndarray,
                     times: np.ndarray = None,
                     original: np.ndarray = None,
                     reconstructed: np.ndarray = None,
                     standardize_windows: List[int] = None,
                     verbose: bool = True) -> Dict:
        
        ### 主要
        rank_ic_results = self.rank_ic(latent_features, target)
        logger.panel(
                f"  Max RankIC: {rank_ic_results['max']:.4f}\n"
                f"  Mean RankIC: {rank_ic_results['mean']:.4f}\n"
                f"  显著特征(>0.01): {rank_ic_results['n_significant_01']}",
                title="[核心] RankIC"
            )
        logger.table(pd.DataFrame([rank_ic_results]).drop(['rank_ics'],axis=1),title="RankIC 分析")


        direction_results = self.direction_accuracy(
            latent_features, target, rank_ic_results['best_feature_idx'])
        logger.panel(
                f"  方向准确率: {direction_results['direction_accuracy']:.1%}\n"
                f"  符号匹配: {direction_results['sign_match']:.1%}\n"
                f"  排序能力: {direction_results['ranking_ability']:.1%}",
                title="[核心] 预测能力"
            )
        
        
        ### 次要
        ic_decay_lags = 4
        best_feat = latent_features[:, rank_ic_results['best_feature_idx']]
        ic_decay_results = self.ic_decay(best_feat, target,  ic_decay_lags)
        decay_str = ' → '.join([f'{v:.3f}' for v in ic_decay_results['decay_curve'][:5]])
        decay_rate = ic_decay_results['decay_rate']
        logger.panel(
                f"  半衰期: {ic_decay_results['value']} \n"
                f"  衰减: {decay_str}"
                f"  衰减幅度： {decay_rate}"
                f"  lag 数: {ic_decay_lags}",
                title="[次要] IC 半衰期"
            )

        
        tail_results = self.tail_prediction(best_feat, target)
        logger.panel(
                f"  高波动 IC: {tail_results['high_vol_ic']:.4f}\n"
                f"  正常 IC: {tail_results['normal_ic']:.4f}\n"
                f"  IC 变化: {-tail_results['ic_drop']:.1%}",
                title="[次要] 尾部预测 (极端行情)"
            )
        
        dead_results = self.dead_neurons(latent_features)
        logger.panel(
                f"  比例: {dead_results['dead_neuron_ratio']:.1%} "
                f"({dead_results['dead_neuron_count']}/{latent_features.shape[1]})",
                title="[核心] 死神经元"
            )
        
        autocorr_results = self.autocorrelation(
            latent_features, rank_ic_results['best_feature_idx'])
        logger.panel(
                f"  Max Lag-1: {autocorr_results['max']:.4f} \n"
                f"  Mean Lag-1: {autocorr_results['mean']:.4f}",
                title="[核心] 自相关"
            )
        
        recon_results = self.reconstruction(original, reconstructed)
        logger.panel(
                    f"  Explained Variance: {recon_results['basic']['explained_variance']:.1%} \n"
                    f"  重建方向准确率: {recon_results['directional']['overall_accuracy']:.1%} "
                    f"(加权: {recon_results['directional']['weighted_accuracy']:.1%}) \n"
                    f"  最后时间步 EV: {recon_results['last_step']['ev']:.1%} \n"
                    f"  相关结构保留: {recon_results['last_step']['correlation_preserved']:.1%}",
                    title="[重要] 重建质量"
                )

        eff_dim_results = self.effective_dimensions(
            latent_features, dead_results['dead_indices'])
        
        logger.panel(
                f"  95%方差: {eff_dim_results['dims_95']}/{eff_dim_results['total']} "
                f"(利用率 {eff_dim_results['utilization']:.1%})",
                title="[辅助] 有效维度"
            )
        
        mi_results = self.mutual_information(latent_features, target, original)
        info_text = f"  Total MI: {mi_results['latent_mi']['total_mi']:.4f}\n"
        info_text += f"  Mean MI: {mi_results['latent_mi']['mean_mi']:.4f}\n"
        info_text += f"  Max MI: {mi_results['latent_mi']['max_mi']:.4f}"
        if 'comparison' in mi_results:
            info_text += f"\n  信息效率: {mi_results['comparison']['information_efficiency']:.2f}"
            info_text += f" ({'✅' if mi_results['comparison']['info_preserved'] else '⚠️'})"
        logger.panel(info_text, title="[辅助] 互信息")


        if standardize_windows:
            std_results = []
            # 添加原始数据结果作为基准
            std_results.append({
                'Window': 'Raw',
                'Max RankIC': rank_ic_results['max'],
                'Mean RankIC': rank_ic_results['mean'],
                'Dir Acc': direction_results['direction_accuracy'],
                'Sign Match': direction_results['sign_match']
            })
            for win in standardize_windows:
                std_features = self._standardize(latent_features, win)
                valid_idx = slice(win-1, None)
                valid_features = std_features[valid_idx]
                valid_target = target[valid_idx]
                
                if len(valid_target) < 100:
                    logger.print(f"⚠️ Window {win} too large for data length {len(target)}, skipping.")
                    continue

                # 3. 评估
                # RankIC
                std_rank_ic = self.rank_ic(valid_features, valid_target)
                # Direction Accuracy (使用原始的最佳特征索引，或者重新计算？通常关注同一个特征的表现变化)
                # 这里我们重新计算最佳特征，看看标准化是否改变了最佳特征
                # 或者为了对比，我们应该看整体能力。
                # 让我们记录 Max RankIC 对应的特征的表现
                std_dir_acc = self.direction_accuracy(
                    valid_features, valid_target, std_rank_ic['best_feature_idx'])
                
                # 4. 额外指标: 自相关 (稳定性) 和 峰度 (异常值处理能力)
                best_feat_std = valid_features[:, std_rank_ic['best_feature_idx']]

                if np.var(best_feat_std) > 1e-8:
                    std_autocorr = np.corrcoef(best_feat_std[:-1], best_feat_std[1:])[0, 1]
                else:
                    std_autocorr = 0.0
                    
                # Kurtosis (Fisher definition, normal=0)
                std_kurtosis = kurtosis(best_feat_std)

                std_results.append({
                    'Window': f'Win={win}',
                    'Max RankIC': std_rank_ic['max'],
                    'Mean RankIC': std_rank_ic['mean'],
                    'Dir Acc': std_dir_acc['direction_accuracy'],
                    'Sign Match': std_dir_acc['sign_match'],
                    'AutoCorr': std_autocorr,
                    'Kurtosis': std_kurtosis
                })

             # 输出对比表格
            logger.panel("标准化对预测能力的影响分析", title="[分析] 标准化影响")
            std_df = pd.DataFrame(std_results)
            # 格式化列
            # 手动格式化列
            std_df['Max RankIC'] = std_df['Max RankIC'].apply(lambda x: f'{x:.4f}')
            std_df['Mean RankIC'] = std_df['Mean RankIC'].apply(lambda x: f'{x:.4f}')
            std_df['Dir Acc'] = std_df['Dir Acc'].apply(lambda x: f'{x:.1%}')
            std_df['Sign Match'] = std_df['Sign Match'].apply(lambda x: f'{x:.1%}')
            std_df['AutoCorr'] = std_df['AutoCorr'].apply(lambda x: f'{x:.4f}')
            std_df['Kurtosis'] = std_df['Kurtosis'].apply(lambda x: f'{x:.2f}')

            logger.table(std_df, title="标准化窗口对比")   