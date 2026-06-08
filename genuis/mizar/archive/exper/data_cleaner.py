"""
数据清洗模块

负责数据清洗工作，包括：
1. 处理缺失值
2. 删除高缺失率特征
3. 删除零方差特征
4. 时间排序
5. 处理多合约数据
"""

import pandas as pd
from typing import Tuple, List
try:
    from . import config
except ImportError:
    import config


class DataCleaner:
    """
    数据清洗器类
    
    提供完整的数据清洗流程，确保数据质量。
    """
    
    def __init__(self, nan_threshold: float = None, 
                 var_threshold: float = None,
                 target_col: str = None):
        """
        初始化数据清洗器
        
        参数:
            nan_threshold: NaN缺失率阈值（超过此值的特征将被删除）
            var_threshold: 方差阈值（小于此值的特征将被删除）
            target_col: 目标变量列名
        """
        self.nan_threshold = nan_threshold if nan_threshold is not None else config.NAN_THRESHOLD
        self.var_threshold = var_threshold if var_threshold is not None else config.VAR_THRESHOLD
        self.target_col = target_col if target_col is not None else config.TARGET_COL
        self.exclude_cols = ['trade_time', 'code', self.target_col]
    
    def analyze_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        分析缺失值情况
        
        参数:
            df: 数据框
        
        返回:
            DataFrame: NaN统计信息
        """
        print("\n[3.0] 分析缺失值情况")
        print("-" * 40)
        
        # 排除时间和code列
        feature_cols_all = [col for col in df.columns 
                           if col not in self.exclude_cols]
        
        nan_stats = pd.DataFrame({
            'feature': feature_cols_all,
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
        """
        删除目标变量为NaN的行
        
        参数:
            df: 数据框
        
        返回:
            DataFrame: 清洗后的数据框
        """
        print("\n[3.1] 处理目标变量缺失值")
        print("-" * 40)
        
        before_len = len(df)
        df = df.dropna(subset=[self.target_col])
        after_len = len(df)
        
        print(f"  删除目标变量NaN: {before_len:,} → {after_len:,} "
              f"(删除{before_len - after_len:,}行)")
        
        return df
    
    def remove_high_nan_features(self, df: pd.DataFrame, 
                                 nan_stats: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        删除高缺失率特征
        
        参数:
            df: 数据框
            nan_stats: NaN统计信息
        
        返回:
            Tuple[DataFrame, List[str]]: 清洗后的数据框和删除的特征列表
        """
        print("\n[3.2] 删除高缺失率特征")
        print("-" * 40)
        print(f"  阈值: NaN比例 > {self.nan_threshold*100}%")
        
        high_nan_cols = nan_stats[nan_stats['nan_ratio'] > self.nan_threshold]['feature'].tolist()
        print(f"  发现 {len(high_nan_cols)} 个高缺失率特征")
        
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
    
    def remove_remaining_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        删除剩余包含NaN的行
        
        参数:
            df: 数据框
        
        返回:
            DataFrame: 清洗后的数据框
        """
        print("\n[3.3] 删除剩余NaN行")
        print("-" * 40)
        
        before_len = len(df)
        df = df.dropna()
        after_len = len(df)
        
        print(f"  删除包含NaN的行: {before_len:,} → {after_len:,} "
              f"(删除{before_len - after_len:,}行)")
        
        print(f"\n  【说明】因子数据通常不适合填充NaN，因为：")
        print(f"    1. 填充会引入虚假信号")
        print(f"    2. 技术指标的NaN往往表示无法计算（如窗口期初）")
        print(f"    3. 删除NaN更保守但更可靠")
        
        return df
    
    def handle_multiple_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理多合约数据（如果有多个合约，保留主力合约）
        
        参数:
            df: 数据框
        
        返回:
            DataFrame: 处理后的数据框
        """
        print("\n[3.4] 处理多合约")
        print("-" * 40)
        
        if 'code' not in df.columns:
            print("  无code列，跳过此步骤")
            return df
        
        n_codes = df['code'].nunique()
        print(f"  唯一合约数: {n_codes}")
        
        if n_codes > 1:
            print(f"  合约分布:")
            code_counts = df['code'].value_counts()
            for code, count in code_counts.items():
                print(f"    {code}: {count:,} 条")
            
            main_code = code_counts.index[0]
            print(f"\n  保留主力合约: {main_code}")
            df = df[df['code'] == main_code].copy()
            print(f"  剩余样本数: {len(df):,}")
        else:
            print(f"  只有单一合约，无需处理")
        
        return df
    
    def sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按时间排序（时间序列预测必须按时间排序）
        
        参数:
            df: 数据框
        
        返回:
            DataFrame: 排序后的数据框
        """
        print("\n[3.5] 时间排序")
        print("-" * 40)
        print(f"  【重要】时间序列预测必须按时间排序！")
        
        # 确保trade_time是日期时间类型
        if not pd.api.types.is_datetime64_any_dtype(df['trade_time']):
            df['trade_time'] = pd.to_datetime(df['trade_time'])
        
        df = df.sort_values('trade_time').reset_index(drop=True)
        print(f"  ✓ 已按时间升序排序")
        print(f"  时间范围: {df['trade_time'].min()} 至 {df['trade_time'].max()}")
        
        return df
    
    def remove_zero_variance_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        删除零方差特征
        
        参数:
            df: 数据框
        
        返回:
            Tuple[DataFrame, List[str]]: 清洗后的数据框和删除的特征列表
        """
        print("\n[3.6] 删除零方差特征")
        print("-" * 40)
        print(f"  【说明】方差为0的特征没有区分能力，应删除")
        
        feature_cols = [col for col in df.columns 
                       if col not in self.exclude_cols]
        
        feature_vars = df[feature_cols].var()
        zero_var_features = feature_vars[feature_vars < self.var_threshold].index.tolist()
        
        print(f"  方差阈值: {self.var_threshold}")
        print(f"  零方差特征数: {len(zero_var_features)}")
        
        if len(zero_var_features) > 0:
            print(f"  删除特征示例:")
            for i, feat in enumerate(zero_var_features[:5], 1):
                print(f"    {i}. {feat}")
            if len(zero_var_features) > 5:
                print(f"    ... (共{len(zero_var_features)}个)")
            
            df = df.drop(columns=zero_var_features)
            print(f"  ✓ 删除完成")
        
        return df, zero_var_features
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的数据清洗流程
        
        参数:
            df: 原始数据框
        
        返回:
            DataFrame: 清洗后的数据框
        """
        print("\n" + "=" * 80)
        print("第3步：数据清洗")
        print("=" * 80)
        
        print("\n【目的】去除无效数据，确保数据质量")
        
        original_shape = df.shape
        print(f"清洗前数据: {original_shape}")
        
        # 1. 分析缺失值
        nan_stats = self.analyze_nan(df)
        
        # 2. 删除目标变量为NaN的行
        df = self.remove_target_nan(df)
        
        # 3. 删除高缺失率特征
        df, high_nan_cols = self.remove_high_nan_features(df, nan_stats)
        
        # 4. 删除剩余NaN行
        df = self.remove_remaining_nan(df)
        
        # 5. 处理多合约
        df = self.handle_multiple_codes(df)
        
        # 6. 时间排序
        df = self.sort_by_time(df)
        
        # 7. 删除零方差特征
        df, zero_var_features = self.remove_zero_variance_features(df)
        
        # 8. 清洗总结
        print("\n[3.7] 清洗总结")
        print("-" * 40)
        cleaned_shape = df.shape
        print(f"  清洗前: {original_shape}")
        print(f"  清洗后: {cleaned_shape}")
        print(f"  样本保留率: {cleaned_shape[0]/original_shape[0]*100:.1f}%")
        print(f"  特征保留率: {cleaned_shape[1]/original_shape[1]*100:.1f}%")
        
        return df

