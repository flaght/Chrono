import pdb
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import acf
from lib import logger

class Evaluator(object):

    def __init__(self, selected_features, target_col):
        self.selected_features = selected_features
        self.target_col = target_col

    def check_dead_neurons(self, latent, threshold=1e-6):
        variances = np.var(latent, axis=0)
        dead_indices = np.where(variances < threshold)[0]
        dead_ratio = len(dead_indices) / latent.shape[1]
        return {
            "dead_neuron_count": len(dead_indices),
            "dead_neuron_ratio": dead_ratio,
            "dead_indices": dead_indices.tolist()
        }

        
    def final_metrics(self, factors_data):
        X = factors_data[self.selected_features].values
        y = factors_data[self.target_col].values

        logger.panel(
            f"  Samples: {len(factors_data)}\n"
            f"  Features: {len(self.selected_features)}\n"
            f"  Target: {self.target_col}",title="Data Prepared")
        
        
        dead_metrics = self.check_dead_neurons(latent=X)
        valid_indices = [i for i in range(X.shape[1]) if i not in dead_metrics['dead_indices']]
        X_valid = X[:, valid_indices]
        valid_feature_names = [self.selected_features[i] for i in valid_indices]
        logger.panel(f"  已移除 {dead_metrics['dead_neuron_count']} 个死神经元。有效特征：{len(valid_indices)}",
                     title="死神经元检测")

        

        rank_ics = []
        for i in range(X_valid.shape[1]):
            ic, _ = spearmanr(X_valid[:, i], y)
            rank_ics.append(abs(ic)) # Use abs IC
        
        top_k_indices = np.argsort(rank_ics)[-10:][::-1]
        top_features = X_valid[:, top_k_indices]
        top_feature_names = [valid_feature_names[i] for i in top_k_indices]
        logger.panel(f"排名前三的特征 RankIC: {rank_ics[top_k_indices[0]]:.4f}, {rank_ics[top_k_indices[1]]:.4f}, {rank_ics[top_k_indices[2]]:.4f}",
                     title="特征 RankIC")
        
        pdb.set_trace()
        times = factors_data['trade_time'].dt.time
        unique_times = sorted(factors_data['trade_time'].dt.time.unique())
        top_feat = top_features[:, 0]
        intraday_pattern = pd.DataFrame({'time': times})
        intraday_pattern['activation'] = np.abs(top_feat)
        pattern = intraday_pattern.groupby('time')['activation'].mean()

        logger.panel(f" 最佳特征的日内模式（{top_feature_names[0]}）",title="日内模式分析")
        sample_times = [unique_times[i] for i in np.linspace(0, len(unique_times)-1, 5).astype(int)]
        log_res = []
        for t in sample_times:
            val = pattern.get(t, 0)
            log_res.append({"trade_time":t, "val":val})
        logger.table(data=pd.DataFrame(log_res),title="举几个例子")


        ### 尾部预测（极端市场状况）
        volatility = np.abs(y)
        high_vol_threshold = np.percentile(volatility, 90)
        high_vol_mask = volatility > high_vol_threshold
        low_vol_mask = volatility <= high_vol_threshold

        # 评估高波动性和低波动性下的 IC
        high_vol_ic, _ = spearmanr(top_feat[high_vol_mask], y[high_vol_mask])
        low_vol_ic, _ = spearmanr(top_feat[low_vol_mask], y[low_vol_mask])

        logger.panel(
        f"  Overall RankIC: {rank_ics[top_k_indices[0]]:.4f}\n"
        f"  High Volatility IC (Top 10%): {abs(high_vol_ic):.4f}\n"
        f"  Normal Market IC: {abs(low_vol_ic):.4f}\n"
        f"  Ratio (High/Normal): {abs(high_vol_ic)/abs(low_vol_ic):.2f}",
        title="Regime-Conditional 预测能力")

        if abs(high_vol_ic) > abs(low_vol_ic) * 1.2:
            logger.print("  ✅ 该特性是“危机阿尔法”（在极端市场中表现更强）。.")
        else:
            logger.print("  ⚠️ 在极端市场环境下，功能性能可能会下降或保持稳定。")

        decay_ics = []
        for lag in range(10):
            # Shift feature
            feat_lagged = np.roll(top_feat, lag)
            # Truncate invalid parts
            valid_mask = slice(lag, None)
            ic, _ = spearmanr(feat_lagged[valid_mask], y[valid_mask])
            decay_ics.append(abs(ic))

        half_life = next((i for i, v in enumerate(decay_ics) if v < decay_ics[0]*0.5), 10)
        decay_str = " -> ".join([f"{v:.3f}" for v in decay_ics[:5]])
        logger.panel(
            f"  IC衰减（特征滞后与目标）:"
            f"    {decay_str} ..."
            f"  估计信息半衰期：{half_life} 步",
            title="IC衰减分析（模拟）"
            )
        
        def estimate_entropy(features):
            # Standardize
            features_std = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            # Covariance
            cov = np.cov(features_std.T)
            # Log determinant
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                return 0 
            d = features.shape[1]
            return 0.5 * (logdet / d + np.log(2 * np.pi * np.e))
    
        if X_valid.shape[1] > 0:
            latent_entropy = estimate_entropy(X_valid)
            logger.print(f"  潜在空间熵（每维）：{latent_entropy:.4f}")
        
            if latent_entropy < -1.0:
                logger.print("  ⚠️ 低熵：特征可能已经坍缩或高度相关。")
            elif latent_entropy > 1.0:
                logger.print("  ✅ High Entropy: Features are rich and diverse.")
        else:
            logger.print("  ⚠️ 高熵：特征丰富多样。")