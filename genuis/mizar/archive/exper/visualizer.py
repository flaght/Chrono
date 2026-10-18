"""
可视化模块

负责生成各种可视化图表，包括：
1. 预测vs实际散点图
2. 残差分布
3. 策略累计收益曲线
4. 回撤曲线
5. 预测分位数收益
6. 方向预测混淆矩阵
7. 特征重要性
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional
try:
    from . import config
except ImportError:
    import config

# 设置绘图参数
plt.rcParams['figure.figsize'] = config.FIG_SIZE
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


class Visualizer:
    """
    可视化器类
    
    提供各种模型评估和结果可视化功能。
    """
    
    def __init__(self, output_dir: str = None):
        """
        初始化可视化器
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir if output_dir is not None else config.OUTPUT_DIR
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  ic: float, direction_acc: float,
                                  save_path: Optional[str] = None):
        """
        绘制预测vs实际散点图
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            ic: IC值
            direction_acc: 方向准确率
            save_path: 保存路径（如果为None，不保存）
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_true, y_pred, alpha=0.3, s=5)
        ax.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()], 'r--', lw=2, label='完美预测线')
        ax.set_xlabel('实际收益率', fontsize=12)
        ax.set_ylabel('预测收益率', fontsize=12)
        ax.set_title(f'预测 vs 实际 (测试集)\nIC={ic:.4f}, 方向准确率={direction_acc:.2%}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      rmse: float, save_path: Optional[str] = None):
        """
        绘制残差分布直方图
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            rmse: RMSE值
            save_path: 保存路径（如果为None，不保存）
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        residuals = y_true - y_pred
        ax.hist(residuals, bins=100, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', lw=2, label='零残差')
        ax.set_xlabel('残差 (实际 - 预测)', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title(f'残差分布\n均值={residuals.mean():.6f}, RMSE={rmse:.6f}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_cumulative_returns(self, strategy_returns: np.ndarray,
                               sharpe: float, save_path: Optional[str] = None):
        """
        绘制策略累计收益曲线
        
        参数:
            strategy_returns: 策略收益序列
            sharpe: Sharpe Ratio
            save_path: 保存路径（如果为None，不保存）
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cumulative_returns = np.cumsum(strategy_returns)
        ax.plot(range(len(cumulative_returns)), cumulative_returns,
               linewidth=1.5, color='blue', label='策略收益')
        ax.axhline(0, color='r', linestyle='--', lw=1, label='盈亏平衡线')
        ax.fill_between(range(len(cumulative_returns)), 0, cumulative_returns,
                       alpha=0.3)
        ax.set_xlabel('交易次数', fontsize=12)
        ax.set_ylabel('累计收益', fontsize=12)
        ax.set_title(f'策略累计收益曲线\n总收益={strategy_returns.sum():.6f}, Sharpe={sharpe:.2f}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_drawdown(self, drawdown: np.ndarray, max_dd: float,
                     save_path: Optional[str] = None):
        """
        绘制策略回撤曲线
        
        参数:
            drawdown: 回撤序列
            max_dd: 最大回撤
            save_path: 保存路径（如果为None，不保存）
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.fill_between(range(len(drawdown)), drawdown, 0,
                       alpha=0.5, color='red', label='回撤区域')
        ax.plot(drawdown, linewidth=1.5, color='darkred', label='回撤曲线')
        ax.axhline(max_dd, color='black', linestyle='--', lw=1,
                  label=f'最大回撤={max_dd:.6f}')
        ax.set_xlabel('交易次数', fontsize=12)
        ax.set_ylabel('回撤', fontsize=12)
        ax.set_title(f'策略回撤曲线\n最大回撤={max_dd:.6f}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_quantile_returns(self, quantile_stats: pd.DataFrame,
                             q5_q1_diff: float, save_path: Optional[str] = None):
        """
        绘制预测分位数收益柱状图
        
        参数:
            quantile_stats: 分位数统计结果
            q5_q1_diff: Q5-Q1差值
            save_path: 保存路径（如果为None，不保存）
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        quantile_means = quantile_stats['mean']
        colors = ['red' if x < 0 else 'green' for x in quantile_means]
        bars = ax.bar(range(len(quantile_means)), quantile_means,
                     alpha=0.7, edgecolor='black', color=colors)
        ax.axhline(0, color='black', linestyle='-', lw=1)
        ax.set_xticks(range(len(quantile_means)))
        ax.set_xticklabels(quantile_means.index, rotation=15)
        ax.set_xlabel('预测值分位数', fontsize=12)
        ax.set_ylabel('平均实际收益', fontsize=12)
        ax.set_title(f'预测分位数 vs 实际收益\nQ5-Q1差值={q5_q1_diff:.6f}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, direction_acc: float,
                             save_path: Optional[str] = None):
        """
        绘制方向预测混淆矩阵
        
        参数:
            cm: 混淆矩阵
            direction_acc: 方向准确率
            save_path: 保存路径（如果为None，不保存）
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, cmap='Blues', alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['预测下跌', '预测上涨'], fontsize=11)
        ax.set_yticklabels(['实际下跌', '实际上涨'], fontsize=11)
        ax.set_title(f'方向预测混淆矩阵\n准确率={direction_acc:.2%}',
                    fontsize=14, fontweight='bold')
        
        # 在每个格子上显示数值
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i,
                             f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                             ha="center", va="center",
                             color="white" if cm[i, j] > cm.max()/2 else "black",
                             fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                               top_n: int = None,
                               save_path: Optional[str] = None):
        """
        绘制特征重要性
        
        参数:
            feature_importance: 特征重要性DataFrame（包含'feature'和'importance_gain'列）
            top_n: 显示前N个特征
            save_path: 保存路径（如果为None，不保存）
        """
        if top_n is None:
            top_n = config.TOP_N_FEATURES
        fig, ax = plt.subplots(figsize=(12, 10))
        
        top_features = feature_importance.head(top_n)
        
        # 截断过长的特征名
        feature_names_short = [
            name[:60] + '...' if len(name) > 60 else name
            for name in top_features['feature']
        ]
        
        y_pos = range(len(top_features))
        ax.barh(y_pos, top_features['importance_gain'], alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names_short, fontsize=9)
        ax.set_xlabel('重要性 (Gain)', fontsize=12)
        ax.set_title(f'Top {top_n} 特征重要性', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, v in enumerate(top_features['importance_gain']):
            ax.text(v, i, f' {v:.0f}', va='center', fontsize=8)
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_evaluation_summary(self, y_test: np.ndarray, y_test_pred: np.ndarray,
                               test_strategy_returns: np.ndarray,
                               test_drawdown: np.ndarray,
                               quantile_stats: pd.DataFrame,
                               cm: np.ndarray,
                               test_ic: float, test_direction_acc: float,
                               test_rmse: float, test_sharpe: float,
                               test_max_dd: float, q5_q1_diff: float,
                               save_path: Optional[str] = None):
        """
        生成完整的评估图表（6张子图）
        
        参数:
            y_test: 测试集真实值
            y_test_pred: 测试集预测值
            test_strategy_returns: 测试集策略收益
            test_drawdown: 测试集回撤序列
            quantile_stats: 分位数统计
            cm: 混淆矩阵
            test_ic: 测试集IC
            test_direction_acc: 测试集方向准确率
            test_rmse: 测试集RMSE
            test_sharpe: 测试集Sharpe Ratio
            test_max_dd: 测试集最大回撤
            q5_q1_diff: Q5-Q1差值
            save_path: 保存路径
        """
        print("\n生成6张可视化图表...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 图1: 预测vs实际（散点图）
        print("  1/6 预测vs实际散点图...")
        axes[0, 0].scatter(y_test, y_test_pred, alpha=0.3, s=5)
        axes[0, 0].plot([y_test.min(), y_test.max()],
                       [y_test.min(), y_test.max()], 'r--', lw=2, label='完美预测线')
        axes[0, 0].set_xlabel('实际收益率', fontsize=12)
        axes[0, 0].set_ylabel('预测收益率', fontsize=12)
        axes[0, 0].set_title(f'预测 vs 实际 (测试集)\nIC={test_ic:.4f}, 方向准确率={test_direction_acc:.2%}',
                           fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 图2: 残差分布
        print("  2/6 残差分布直方图...")
        residuals = y_test - y_test_pred
        axes[0, 1].hist(residuals, bins=100, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(0, color='r', linestyle='--', lw=2, label='零残差')
        axes[0, 1].set_xlabel('残差 (实际 - 预测)', fontsize=12)
        axes[0, 1].set_ylabel('频数', fontsize=12)
        axes[0, 1].set_title(f'残差分布\n均值={residuals.mean():.6f}, RMSE={test_rmse:.6f}',
                           fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 图3: 策略累计收益曲线
        print("  3/6 策略累计收益曲线...")
        cumulative_returns = np.cumsum(test_strategy_returns)
        axes[0, 2].plot(range(len(cumulative_returns)), cumulative_returns,
                       linewidth=1.5, color='blue', label='策略收益')
        axes[0, 2].axhline(0, color='r', linestyle='--', lw=1, label='盈亏平衡线')
        axes[0, 2].fill_between(range(len(cumulative_returns)), 0, cumulative_returns,
                               alpha=0.3)
        axes[0, 2].set_xlabel('交易次数', fontsize=12)
        axes[0, 2].set_ylabel('累计收益', fontsize=12)
        axes[0, 2].set_title(f'策略累计收益曲线\n总收益={test_strategy_returns.sum():.6f}, Sharpe={test_sharpe:.2f}',
                           fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 图4: 回撤曲线
        print("  4/6 策略回撤曲线...")
        axes[1, 0].fill_between(range(len(test_drawdown)), test_drawdown, 0,
                               alpha=0.5, color='red', label='回撤区域')
        axes[1, 0].plot(test_drawdown, linewidth=1.5, color='darkred', label='回撤曲线')
        axes[1, 0].axhline(test_max_dd, color='black', linestyle='--', lw=1,
                          label=f'最大回撤={test_max_dd:.6f}')
        axes[1, 0].set_xlabel('交易次数', fontsize=12)
        axes[1, 0].set_ylabel('回撤', fontsize=12)
        axes[1, 0].set_title(f'策略回撤曲线\n最大回撤={test_max_dd:.6f}',
                           fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 图5: 预测分位数收益
        print("  5/6 预测分位数收益柱状图...")
        quantile_means = quantile_stats['mean']
        colors = ['red' if x < 0 else 'green' for x in quantile_means]
        bars = axes[1, 1].bar(range(len(quantile_means)), quantile_means,
                             alpha=0.7, edgecolor='black', color=colors)
        axes[1, 1].axhline(0, color='black', linestyle='-', lw=1)
        axes[1, 1].set_xticks(range(len(quantile_means)))
        axes[1, 1].set_xticklabels(quantile_means.index, rotation=15)
        axes[1, 1].set_xlabel('预测值分位数', fontsize=12)
        axes[1, 1].set_ylabel('平均实际收益', fontsize=12)
        axes[1, 1].set_title(f'预测分位数 vs 实际收益\nQ5-Q1差值={q5_q1_diff:.6f}',
                           fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 图6: 方向预测混淆矩阵
        print("  6/6 方向预测混淆矩阵...")
        im = axes[1, 2].imshow(cm, cmap='Blues', alpha=0.8)
        axes[1, 2].set_xticks([0, 1])
        axes[1, 2].set_yticks([0, 1])
        axes[1, 2].set_xticklabels(['预测下跌', '预测上涨'], fontsize=11)
        axes[1, 2].set_yticklabels(['实际下跌', '实际上涨'], fontsize=11)
        axes[1, 2].set_title(f'方向预测混淆矩阵\n准确率={test_direction_acc:.2%}',
                           fontsize=14, fontweight='bold')
        
        # 在每个格子上显示数值
        for i in range(2):
            for j in range(2):
                text = axes[1, 2].text(j, i,
                                     f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                                     ha="center", va="center",
                                     color="white" if cm[i, j] > cm.max()/2 else "black",
                                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            print(f"\n✓ 评估图表已保存至: {save_path}")
            plt.close()
        else:
            plt.show()

