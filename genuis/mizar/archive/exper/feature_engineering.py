"""
特征工程模块

负责特征筛选和工程工作，包括：
1. 计算因子IC（Information Coefficient）
2. 智能特征筛选（基于相关性和IC）
3. 特征选择
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
try:
    from . import config
except ImportError:
    import config


class FeatureEngineer:
    """
    特征工程器类
    
    提供特征筛选和工程功能。
    """
    
    def __init__(self, corr_threshold: float = None,
                 ic_threshold: float = None,
                 target_col: str = None):
        """
        初始化特征工程器
        
        参数:
            corr_threshold: 相关性阈值（超过此值的特征对将删除IC较低的那个）
            ic_threshold: IC阈值（小于此值的特征将被删除）
            target_col: 目标变量列名
        """
        self.corr_threshold = corr_threshold if corr_threshold is not None else config.CORR_THRESHOLD
        self.ic_threshold = ic_threshold if ic_threshold is not None else config.IC_THRESHOLD
        self.target_col = target_col if target_col is not None else config.TARGET_COL
        self.exclude_cols = ['trade_time', 'code', self.target_col]
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        获取特征列列表（排除时间和code列）
        
        参数:
            df: 数据框
        
        返回:
            List[str]: 特征列名列表
        """
        return [col for col in df.columns if col not in self.exclude_cols]
    
    def calculate_ic(self, df: pd.DataFrame, 
                    feature_cols: List[str]) -> Dict[str, float]:
        """
        计算因子IC（Information Coefficient）
        
        IC是因子与目标变量的相关系数，衡量因子的预测能力。
        IC越高，因子预测能力越强。
        
        参数:
            df: 数据框
            feature_cols: 特征列列表
        
        返回:
            Dict[str, float]: 特征IC字典
        """
        print("\n[4.3] 计算因子IC（Information Coefficient）")
        print("-" * 40)
        print(f"  【说明】IC是因子与目标变量的相关系数，衡量因子的预测能力")
        print(f"  IC越高，因子预测能力越强")
        
        ic_dict = {}
        print(f"  计算中...")
        
        for i, col in enumerate(feature_cols, 1):
            if i % 50 == 0:
                print(f"    进度: {i}/{len(feature_cols)}")
            
            try:
                ic = df[col].corr(df[self.target_col])
                # 使用绝对值，因为正相关和负相关都有预测价值
                ic_dict[col] = abs(ic) if not np.isnan(ic) else 0
            except:
                ic_dict[col] = 0
        
        ic_series = pd.Series(ic_dict).sort_values(ascending=False)
        
        print(f"\n  IC统计:")
        print(f"    平均IC: {ic_series.mean():.4f}")
        print(f"    中位IC: {ic_series.median():.4f}")
        print(f"    最大IC: {ic_series.max():.4f}")
        print(f"    IC>0.01的因子数: {(ic_series > 0.01).sum()}")
        print(f"    IC>0.03的因子数: {(ic_series > 0.03).sum()}")
        
        print(f"\n  Top 20 高IC因子:")
        print(ic_series.head(20).to_frame('IC'))
        
        return ic_dict
    
    def smart_feature_selection(self, df: pd.DataFrame,
                               feature_cols: List[str],
                               ic_dict: Dict[str, float]) -> List[str]:
        """
        智能特征筛选
        
        筛选策略：
        1. 对于高度相关的因子对，保留IC更高的
        2. 删除IC过低（无预测能力）的因子
        
        参数:
            df: 数据框
            feature_cols: 特征列列表
            ic_dict: 特征IC字典
        
        返回:
            List[str]: 筛选后的特征列表
        """
        print("\n[4.4] 智能特征筛选")
        print("-" * 40)
        print(f"  【说明】基于相关性和IC进行特征筛选")
        print(f"  筛选策略:")
        print(f"    1. 对于高度相关的因子对，保留IC更高的")
        print(f"    2. 删除IC过低（无预测能力）的因子")
        
        print(f"\n  参数设置:")
        print(f"    相关性阈值: {self.corr_threshold}")
        print(f"    IC阈值: {self.ic_threshold}")
        
        X = df[feature_cols]
        
        # 步骤1: 计算相关性矩阵
        print(f"\n  步骤1: 计算特征相关性矩阵...")
        print(f"    矩阵大小: {len(feature_cols)} × {len(feature_cols)}")
        corr_matrix = X.corr().abs()
        
        # 步骤2: 识别高相关特征对
        print(f"\n  步骤2: 识别高相关特征对...")
        upper_tri = np.triu(corr_matrix.values, k=1)
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if upper_tri[i, j] > self.corr_threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        upper_tri[i, j]
                    ))
        
        print(f"    发现 {len(high_corr_pairs)} 个高相关特征对（相关性>{self.corr_threshold}）")
        
        # 步骤3: 基于IC筛选高相关特征
        print(f"\n  步骤3: 基于IC筛选高相关特征...")
        to_drop = set()
        
        for col1, col2, corr_val in high_corr_pairs:
            # 保留IC高的因子
            if ic_dict[col1] < ic_dict[col2]:
                to_drop.add(col1)
            else:
                to_drop.add(col2)
        
        print(f"    删除 {len(to_drop)} 个低IC特征")
        
        remaining_features = [f for f in feature_cols if f not in to_drop]
        
        # 步骤4: 删除低IC因子
        print(f"\n  步骤4: 删除低IC因子...")
        low_ic_features = [f for f in remaining_features 
                          if ic_dict[f] < self.ic_threshold]
        print(f"    删除 {len(low_ic_features)} 个IC<{self.ic_threshold}的因子")
        
        remaining_features = [f for f in remaining_features 
                            if f not in low_ic_features]
        
        return remaining_features
    
    def select_features(self, df: pd.DataFrame,
                       corr_threshold: float = None,
                       ic_threshold: float = None) -> Tuple[List[str], Dict[str, float]]:
        """
        执行完整的特征选择流程
        
        参数:
            df: 数据框
            corr_threshold: 相关性阈值（如果为None，使用默认值）
            ic_threshold: IC阈值（如果为None，使用默认值）
        
        返回:
            Tuple[List[str], Dict[str, float]]: 选择的特征列表和IC字典
        """
        print("\n" + "=" * 80)
        print("第4步：特征工程")
        print("=" * 80)
        
        print("\n【目的】筛选有效特征，提升模型性能")
        
        # 使用传入的阈值或默认值
        if corr_threshold is not None:
            self.corr_threshold = corr_threshold
        if ic_threshold is not None:
            self.ic_threshold = ic_threshold
        
        # 获取特征列
        print("\n[4.1] 提取特征列")
        print("-" * 40)
        feature_cols = self.get_feature_columns(df)
        print(f"  原始特征数: {len(feature_cols)}")
        
        # 计算IC
        ic_dict = self.calculate_ic(df, feature_cols)
        
        # 智能特征筛选
        selected_features = self.smart_feature_selection(
            df, feature_cols, ic_dict
        )
        
        print(f"\n[特征筛选总结]")
        print(f"  原始特征数: {len(feature_cols)}")
        print(f"  筛选后特征数: {len(selected_features)}")
        print(f"  保留率: {len(selected_features)/len(feature_cols)*100:.1f}%")
        
        return selected_features, ic_dict

