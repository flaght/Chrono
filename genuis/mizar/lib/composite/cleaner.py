import pandas as pd
from typing import Tuple, List
from lib import logger


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

        logger.panel(content=f"  总特征数: {len(feature_cols_all)} \n" 
                  f"  完全无缺失的特征数: {(nan_stats['nan_ratio'] == 0).sum()}\n"
                f"  有缺失的特征数: {(nan_stats['nan_ratio'] > 0).sum()}\n"
                f"  缺失Nan>10%的特征数: {(nan_stats['nan_ratio'] > 0.1).sum()}\n"
                f"  总特征数: {len(feature_cols_all)}", title="特征Nan统计")

        if len(nan_stats) > 0:
            logger.table(nan_stats.head(10),"缺失最严重的前10个特征")

        return nan_stats

    def remove_target_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        before_len = len(df)
        df = df.dropna(subset=[self.target_col])
        after_len = len(df)

        #logger.print(f"  删除目标变量NaN: {before_len:,} → {after_len:,} "
        #            f"(删除{before_len - after_len:,}行)")

        logger.panel(f"  删除目标变量NaN: {before_len:,} → {after_len:,}"
                     f"(删除{before_len - after_len:,}行) \n",
                     "目标变量统计")
        return df

    def remove_high_nan_features(
            self, df: pd.DataFrame,
            nan_stats: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:

        high_nan_cols = nan_stats[nan_stats['nan_ratio'] >
                                  self.nan_threshold]['feature'].tolist()
        if len(high_nan_cols) > 0:
            #logger.print(f"  删除特征:")
            content = ""
            for i, col in enumerate(high_nan_cols[:5], 1):
                nan_ratio = nan_stats[nan_stats['feature'] ==
                                      col]['nan_ratio'].values[0]
                #logger.print(f"    {i}. {col} (NaN比例: {nan_ratio*100:.1f}%)")
                content += f"    {i}. {col} (NaN比例: {nan_ratio*100:.1f}%) \n"
            #if len(high_nan_cols) > 5:
            #    logger.print(f"    ... (共{len(high_nan_cols)}个)")
            content += f"\n    ... (共{len(high_nan_cols)}个)"
            logger.panel(content, f"  删除特征")
            df = df.drop(columns=high_nan_cols)
            logger.print(f"  ✓ 删除完成")

        return df, high_nan_cols

    def sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按时间排序（时间序列预测必须按时间排序）
        
        参数:
            df: 数据框
        
        返回:
            DataFrame: 排序后的数据框
        """
        #logger.print("\n[3.5] 时间排序")
        #logger.print("-" * 40)
        #logger.print(f"  【重要】时间序列预测必须按时间排序！")

        # 确保trade_time是日期时间类型
        if not pd.api.types.is_datetime64_any_dtype(df['trade_time']):
            df['trade_time'] = pd.to_datetime(df['trade_time'])

        df = df.sort_values('trade_time').reset_index(drop=True)
        #logger.print(f"  ✓ 已按时间升序排序")
        #logger.print(
        #    f"  时间范围: {df['trade_time'].min()} 至 {df['trade_time'].max()}")

        logger.panel(f"  【重要】时间序列预测必须按时间排序！\n"
                     f"  ✓ 已按时间升序排序 \n"
                     f"  时间范围: {df['trade_time'].min()} 至 {df['trade_time'].max()}", 
                     "时间排序")
        return df

    def remove_remaining_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        #logger.print("\n[3.3] 删除剩余NaN行")
        #logger.print("-" * 40)

        before_len = len(df)
        df = df.dropna()
        after_len = len(df)

        #logger.print(f"  删除包含NaN的行: {before_len:,} → {after_len:,} "
        #            f"(删除{before_len - after_len:,}行)")

        logger.panel(f"  删除包含NaN的行: {before_len:,} → {after_len:,} \n"
                     f"(删除{before_len - after_len:,}行) \n", "删除剩余NaN行")

        logger.panel(f"    1. 填充会引入虚假信号\n"
                     f"    2. 技术指标的NaN往往表示无法计算（如窗口期初）\n"
                     f"    3. 删除NaN更保守但更可靠", "因子数据通常不适合填充NaN\n")
        #logger.print(f"\n  【说明】因子数据通常不适合填充NaN，因为：")
        #logger.print(f"    1. 填充会引入虚假信号")
        #logger.print(f"    2. 技术指标的NaN往往表示无法计算（如窗口期初）")
        #logger.print(f"    3. 删除NaN更保守但更可靠")

        return df

    def remove_zero_variance_features(
            self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        删除零方差特征
        清洗后的数据框和删除的特征列表
        """
        #logger.print("\n[3.6] 删除零方差特征")
        #logger.print("-" * 40)
        #logger.print(f"  【说明】方差为0的特征没有区分能力，应删除")

        feature_cols = [
            col for col in df.columns if col not in self.exclude_cols
        ]

        feature_vars = df[feature_cols].var()
        zero_var_features = feature_vars[feature_vars <
                                         self.var_threshold].index.tolist()

        #logger.print(f"  方差阈值: {self.var_threshold}")
        #logger.print(f"  零方差特征数: {len(zero_var_features)}")
        content = ""
        if len(zero_var_features) > 0:
            #logger.print(f"  删除特征示例:")
            content += f"  删除特征示例:\n"
            for i, feat in enumerate(zero_var_features[:5], 1):
                #logger.print(f"    {i}. {feat}")
                content += f"    {i}. {feat}\n"
            if len(zero_var_features) > 5:
                #logger.print(f"    ... (共{len(zero_var_features)}个)")
                content += f"    ... (共{len(zero_var_features)}个)\n"

            df = df.drop(columns=zero_var_features)
            #logger.print(f"  ✓ 删除完成")
            content += f"  ✓ 删除完成"

        logger.panel(f"  【说明】方差为0的特征没有区分能力，应删除\n"
                     f"  方差阈值: {self.var_threshold}\n"
                     f"  零方差特征数: {len(zero_var_features)}\n"
                     f"{content}\n","删除零方差特征\n")
        return df, zero_var_features


    def clean(self, df: pd.DataFrame) -> pd.DataFrame:

        original_shape = df.shape
        logger.print(f"清洗前数据: {original_shape}")
        # 1. 分析缺失值
        nan_stats = self.analyze_nan(df)

        # 2. 删除目标变量为NaN的行
        df = self.remove_target_nan(df)

        # 3. 删除高缺失率特征
        df, high_nan_cols = self.remove_high_nan_features(df, nan_stats)

        # 4. 删除剩余NaN行
        df = self.remove_remaining_nan(df)

        # 5. 时间排序
        df = self.sort_by_time(df)

        # 6. 删除零方差特征
        df, zero_var_features = self.remove_zero_variance_features(df)

        cleaned_shape = df.shape
        logger.panel(f"  清洗前: {original_shape} \n"
                     f"  清洗后: {cleaned_shape} \n"
                     f"  样本保留率: {cleaned_shape[0]/original_shape[0]*100:.1f}% \n"
                     f"  特征保留率: {cleaned_shape[1]/original_shape[1]*100:.1f}% \n",
                     "清洗总结")

        return df
