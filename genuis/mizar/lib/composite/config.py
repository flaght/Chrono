import pandas as pd
from typing import Tuple, List


class DataCleaner(object):

    def __init__(self,
                 nan_threshold: float = None,
                 var_threshold: float = None,
                 target_col: str = None):
        """
        初始化数据清洗器
        
        参数:
            nan_threshold: NaN缺失率阈值（超过此值的特征将被删除）
            var_threshold: 方差阈值（小于此值的特征将被删除）
            target_col: 目标变量列名
        """
        self.nan_threshold = nan_threshold
        self.var_threshold = var_threshold
        self.target_col = target_col
        self.exclude_cols = ['trade_time', 'code', self.target_col]

    def analyze_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        # 排除时间和code列
        feature_cols_all = [
            col for col in df.columns if col not in self.exclude_cols
        ]

        nan_stats = pd.DataFrame({
            'feature':
            feature_cols_all,
            'nan_count': [df[col].isna().sum() for col in feature_cols_all],
            'nan_ratio': [df[col].isna().mean() for col in feature_cols_all]
        }).sort_values('nan_ratio', ascending=False)

        print(f"  总特征数: {len(feature_cols_all)}")
        print(f"  完全无缺失的特征数: {(nan_stats['nan_ratio'] == 0).sum()}")
        print(f"  有缺失的特征数: {(nan_stats['nan_ratio'] > 0).sum()}")
        print(f"  缺失>50%的特征数: {(nan_stats['nan_ratio'] > 0.5).sum()}")

        if len(nan_stats) > 0:
            print(f"\n  缺失最严重的前10个特征:")
            print(nan_stats.head(10).to_string(index=False))

        return nan_stats

    def remove_target_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        before_len = len(df)
        df = df.dropna(subset=[self.target_col])
        after_len = len(df)

        print(f"  删除目标变量NaN: {before_len:,} → {after_len:,} "
              f"(删除{before_len - after_len:,}行)")

        return df

    def remove_high_nan_features(
            self, df: pd.DataFrame,
            nan_stats: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:

        high_nan_cols = nan_stats[nan_stats['nan_ratio'] >
                                  self.nan_threshold]['feature'].tolist()
        if len(high_nan_cols) > 0:
            print(f"  删除特征:")
            for i, col in enumerate(high_nan_cols[:5], 1):
                nan_ratio = nan_stats[nan_stats['feature'] == col]['nan_ratio'].values[0]
                print(f"    {i}. {col} (NaN比例: {nan_ratio*100:.1f}%)")
            if len(high_nan_cols) > 5:
                print(f"    ... (共{len(high_nan_cols)}个)")
            
            df = df.drop(columns=high_nan_cols)
            print(f"  ✓ 删除完成")
        
        return df, high_nan_cols

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:

        # 1. 分析缺失值
        nan_stats = self.analyze_nan(df)

        # 2. 删除目标变量为NaN的行
        df = self.remove_target_nan(df)
