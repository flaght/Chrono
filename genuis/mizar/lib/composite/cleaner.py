import pdb
import pandas as pd
import numpy as np
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

        # 计算缺失数据属性
        nan_attributes = self._calculate_nan_attributes(df, feature_cols_all, nan_stats)
        
        # 构建日志内容
        content = self._build_nan_analysis_log(feature_cols_all, nan_stats, nan_attributes)

        logger.panel(content=content, title="特征Nan统计")

        if len(nan_stats) > 0:
            logger.table(nan_stats.head(10),"缺失最严重的前10个特征")

        return nan_stats

    def _calculate_nan_attributes(self, df: pd.DataFrame, feature_cols: list, nan_stats: pd.DataFrame) -> dict:
        """计算缺失数据的各种属性"""
        attributes = {}

        # 基础统计
        attributes['total_features'] = len(feature_cols)
        attributes['complete_features'] = (nan_stats['nan_ratio'] == 0).sum()
        attributes['missing_features'] = (nan_stats['nan_ratio'] > 0).sum()
        attributes['high_missing_features'] = (nan_stats['nan_ratio'] > 0.1).sum()
        attributes['severe_missing_features'] = (nan_stats['nan_ratio'] > 0.5).sum()

        # 缺失值分布统计
        if attributes['missing_features'] > 0:
            missing_ratios = nan_stats[nan_stats['nan_ratio'] > 0]['nan_ratio']
            attributes['missing_ratio_mean'] = missing_ratios.mean()
            attributes['missing_ratio_median'] = missing_ratios.median()
            attributes['missing_ratio_std'] = missing_ratios.std()
            attributes['missing_ratio_skew'] = missing_ratios.skew()

            # 缺失程度分类
            attributes['low_missing'] = ((missing_ratios > 0) & (missing_ratios <= 0.01)).sum()
            attributes['moderate_missing'] = ((missing_ratios > 0.01) & (missing_ratios <= 0.1)).sum()
            attributes['high_missing'] = ((missing_ratios > 0.1) & (missing_ratios <= 0.5)).sum()
            attributes['severe_missing'] = (missing_ratios > 0.5).sum()

        # 时序相关属性（如果有trade_time列）
        #if 'trade_time' in df.columns and attributes['missing_features'] > 0:
        #    attributes.update(self._analyze_temporal_nan_patterns(df, feature_cols))

        # 特征间缺失相关性
        #if attributes['missing_features'] > 1:
        #    attributes.update(self._analyze_nan_correlations(df, feature_cols))

        return attributes

    def _analyze_temporal_nan_patterns(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """分析缺失值的时间模式"""
        temporal_attrs = {}

        # 按时间分组计算缺失率
        if pd.api.types.is_datetime64_any_dtype(df['trade_time']):
            df_temp = df.copy()
            df_temp['date'] = df_temp['trade_time'].dt.date

            # 计算每日缺失统计
            daily_missing = {}
            for col in feature_cols:
                if df[col].isna().any():
                    daily_stats = df_temp.groupby('date')[col].apply(lambda x: x.isna().mean())
                    daily_missing[col] = {
                        'missing_days': (daily_stats > 0).sum(),
                        'max_daily_missing': daily_stats.max(),
                        'avg_daily_missing': daily_stats.mean(),
                        'missing_streak': self._find_longest_missing_streak(df[col])
                    }

            if daily_missing:
                temporal_attrs['temporal_patterns'] = daily_missing
                temporal_attrs['features_with_temporal_patterns'] = len(daily_missing)

                # 整体时间模式
                all_daily_missing_rates = []
                for stats in daily_missing.values():
                    all_daily_missing_rates.extend([stats['max_daily_missing']] * stats['missing_days'])

                if all_daily_missing_rates:
                    temporal_attrs['overall_temporal_volatility'] = pd.Series(all_daily_missing_rates).std()

        return temporal_attrs

    def _find_longest_missing_streak(self, series: pd.Series) -> int:
        """找到最长的连续缺失序列"""
        is_na = series.isna()
        streaks = []
        current_streak = 0

        for val in is_na:
            if val:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0

        if current_streak > 0:
            streaks.append(current_streak)

        return max(streaks) if streaks else 0

    def _analyze_nan_correlations(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """分析特征间缺失值的相关性"""
        corr_attrs = {}

        # 创建缺失指示矩阵
        missing_matrix = df[feature_cols].isna().astype(int)

        # 计算缺失相关性
        if missing_matrix.shape[1] > 1:
            missing_corr = missing_matrix.corr()

            # 找出高度相关的缺失模式
            upper_tri = missing_corr.where(np.triu(np.ones_like(missing_corr), k=1).astype(bool))
            high_missing_corr = upper_tri.stack().abs().sort_values(ascending=False)

            corr_attrs['missing_correlations'] = high_missing_corr[high_missing_corr > 0.7].to_dict()
            corr_attrs['highly_correlated_missing_pairs'] = len(corr_attrs['missing_correlations'])

            # 缺失模式聚类
            #corr_attrs['missing_pattern_clusters'] = self._identify_missing_patterns(missing_matrix)

        return corr_attrs

    def _identify_missing_patterns(self, missing_matrix: pd.DataFrame) -> dict:
        """识别缺失模式聚类"""
        patterns = {}

        # 简单的模式识别：完全同时缺失的特征组
        corr_matrix = missing_matrix.T.corr()

        # 找到相关性>0.7的特征组
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        patterns['highly_synchronized_missing'] = len(high_corr_pairs)
        patterns['missing_pattern_groups'] = high_corr_pairs[:5]  # Top 5

        return patterns

    def _build_nan_analysis_log(self, feature_cols: list, nan_stats: pd.DataFrame, attributes: dict) -> str:
        """构建缺失数据分析的日志内容"""
        content_parts = []

        # 基础统计
        content_parts.append(f"  总特征数: {attributes['total_features']}")
        content_parts.append(f"  完全无缺失的特征数: {attributes['complete_features']}")
        content_parts.append(f"  有缺失的特征数: {attributes['missing_features']}")

        if attributes['missing_features'] > 0:
            content_parts.append(f"  缺失>10%的特征数: {attributes['high_missing_features']}")
            content_parts.append(f"  缺失>50%的特征数: {attributes['severe_missing_features']}")

            # 缺失分布统计
            content_parts.append(f"")
            content_parts.append(f"  缺失分布统计:")
            content_parts.append(f"    平均缺失率: {attributes['missing_ratio_mean']:.4f}")
            content_parts.append(f"    中位缺失率: {attributes['missing_ratio_median']:.4f}")
            content_parts.append(f"    缺失率标准差: {attributes['missing_ratio_std']:.4f}")
            content_parts.append(f"    缺失率偏度: {attributes['missing_ratio_skew']:.4f}")

            # 缺失程度分类
            content_parts.append(f"")
            content_parts.append(f"  缺失程度分类:")
            content_parts.append(f"    轻微缺失(0-1%): {attributes['low_missing']}")
            content_parts.append(f"    中等缺失(1-10%): {attributes['moderate_missing']}")
            content_parts.append(f"    严重缺失(10-50%): {attributes['high_missing']}")
            content_parts.append(f"    极重缺失(>50%): {attributes['severe_missing']}")

            # 时序模式
            if 'temporal_patterns' in attributes:
                content_parts.append(f"")
                content_parts.append(f"  时序缺失模式:")
                content_parts.append(f"    具有时序模式的特征数: {attributes['features_with_temporal_patterns']}")
                content_parts.append(f"    时序波动率: {attributes.get('overall_temporal_volatility', 'N/A'):.4f}")

            # 缺失相关性
            if 'highly_correlated_missing_pairs' in attributes and attributes['highly_correlated_missing_pairs'] > 0:
                content_parts.append(f"")
                content_parts.append(f"  缺失相关性:")
                content_parts.append(f"    高度相关的缺失特征对: {attributes['highly_correlated_missing_pairs']}")
                content_parts.append(f"    同步缺失模式组: {attributes['missing_pattern_clusters']['highly_synchronized_missing']}")

        return "\n".join(content_parts)

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
