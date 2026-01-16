import pdb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from lib import logger
from lib.fa001 import calculate_correlation_matrix

class Featurer(object):
    def __init__(self, corr_threshold: float = None,
                 ic_threshold: float = None,
                 target_col: str = None):
        
        self.corr_threshold = corr_threshold
        self.ic_threshold = ic_threshold
        self.target_col = target_col
        self.exclude_cols = ['trade_time', 'code', self.target_col]

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        return [col for col in df.columns if col not in self.exclude_cols]
    
    def calculate_ic(self, df: pd.DataFrame, 
                    feature_cols: List[str],
                    roll_win=0, 
                    resampling_win=0) -> Dict[str, float]:
        #logger.print("\n计算因子IC（Information Coefficient）")
        #logger.print("-" * 40)
        #logger.print(f"  【说明】IC是因子与目标变量的相关系数，衡量因子的预测能力")
        #logger.print(f"  IC越高，因子预测能力越强")
        content = f"  【说明】IC是因子与目标变量的相关系数，衡量因子的预测能力\n"
        content += f"  IC越高，因子预测能力越强\n"
        ic_dict = {}
        for i, col in enumerate(logger.progress(feature_cols, description="[green]计算因子IC...[/green]"), 1):
            try:
                if roll_win > 0 and resampling_win > 0:
                    df1 = df[['trade_time','code',col, self.target_col]]
                    is_on_mark = df1['trade_time'].dt.minute % int(resampling_win) == 0
                    resample_data = df1[is_on_mark]
                    ic = resample_data[self.target_col].rolling(window=roll_win,min_periods=5).corr(resample_data[col]).mean()

                else:
                    ic = df[col].corr(df[self.target_col])
                
                # 使用绝对值，因为正相关和负相关都有预测价值
                ic_dict[col] = abs(ic) if not np.isnan(ic) and not np.isinf(ic) else 0
            except:
                ic_dict[col] = 0
        ic_series = pd.Series(ic_dict).sort_values(ascending=False)

        #logger.print(f"\n  IC统计:")
        #logger.print(f"    平均IC: {ic_series.mean():.4f}")
        #logger.print(f"    中位IC: {ic_series.median():.4f}")
        #logger.print(f"    最大IC: {ic_series.max():.4f}")
        #logger.print(f"    IC>0.01的因子数: {(ic_series > 0.01).sum()}")
        #logger.print(f"    IC>0.03的因子数: {(ic_series > 0.03).sum()}")
        content += f"\n  IC统计:\n"
        content += f"    平均IC: {ic_series.mean():.4f}\n"
        content += f"    中位IC: {ic_series.median():.4f}\n"
        content += f"    最大IC: {ic_series.max():.4f}\n"
        content += f"    IC>0.01的因子数: {(ic_series > 0.01).sum()}\n"
        content += f"    IC>0.03的因子数: {(ic_series > 0.03).sum()}\n"

        logger.panel(content, "计算因子IC（Information Coefficient）")
        
        logger.table(ic_series.head(20),"Top 20 高IC因子:")
        #logger.info(f"\n  Top 20 高IC因子:")
        #logger.info(ic_series.head(20).to_frame('IC'))
        
        return ic_dict, ic_series
        
    def smart_feature_selection(self, df: pd.DataFrame,
                               feature_cols: List[str],
                               ic_dict: Dict[str, float],
                               method: str = 'custom_ic_correlation',
                               **kwargs) -> List[str]:
        method_descriptions = {
            'factor_values': '全量因子值相关性',
            'rolling_factor_values': '滚动因子值相关性',
            'ic_correlation': '通用时序因子收益率相关性',
            'custom_ic_correlation': '滚动因子收益率相关性'}
        method_name = method_descriptions[method]

        logger.panel(f"  筛选策略:"
                     f"    1. 对于高度相关的因子对，保留IC更高的\n"
                     f"    2. 删除IC过低（无预测能力）的因子\n"
                     f"\n  参数设置:\n"
                     f"    相关性阈值: {self.corr_threshold}\n"
                     f"    IC阈值: {self.ic_threshold}\n","【说明】基于相关性和IC进行特征筛选"
                     f"    基于{method_name}进行特征筛选")
        
        '''
        X = df[feature_cols]

        # 步骤1: 计算相关性矩阵-->此处可替换成，因子收益率相关性
        #logger.print(f"\n  步骤1: 计算特征相关性矩阵...")
        #logger.print(f"    矩阵大小: {len(feature_cols)} × {len(feature_cols)}")
        logger.panel(f"    矩阵大小: {len(feature_cols)} × {len(feature_cols)}",
                     f"计算特征相关性矩阵...")

        corr_matrix = X.corr().abs()
        '''
        corr_matrix = calculate_correlation_matrix(df=df, feature_cols=feature_cols, method=method, **kwargs)


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
        #logger.print(f"    发现 {len(high_corr_pairs)} 个高相关特征对（相关性>{self.corr_threshold}）")
        logger.panel(f"    发现 {len(high_corr_pairs)} 个高相关特征对（相关性>{self.corr_threshold}）","识别高相关特征对...")

        #logger.print(f"\n  步骤3: 基于IC筛选高相关特征...")
        to_drop = set()

        for col1, col2, corr_val in high_corr_pairs:
            # 保留IC高的因子
            if ic_dict[col1] < ic_dict[col2]:
                to_drop.add(col1)
            else:
                to_drop.add(col2)
        
        #logger.print(f"    删除 {len(to_drop)} 个低IC特征")
        logger.panel(f"    从 {len(high_corr_pairs)} 个高相关对中删除 {len(to_drop)} 个IC较低的特征",
             "基于IC筛选高相关特征...")

        remaining_features = [f for f in feature_cols if f not in to_drop]
        
        # 步骤4: 删除低IC因子
        #logger.print(f"\n  步骤4: 删除低IC因子...")
        low_ic_features = [f for f in remaining_features 
                          if ic_dict[f] < self.ic_threshold]
        #logger.print(f"    删除 {len(low_ic_features)} 个IC<{self.ic_threshold}的因子")
        
        logger.panel(f"    删除 {len(low_ic_features)} 个IC<{self.ic_threshold}的因子", "删除低IC因子...")
        remaining_features = [f for f in remaining_features 
                            if f not in low_ic_features]
        
        return remaining_features
        
        
    def select_features(self, df: pd.DataFrame,
                       ic_threshold: float = None,
                       method: str = None,
                       roll_win: int = 0,
                       resampling_win: int = 0) -> Tuple[List[str], Dict[str, float]]:
        
        logger.rule(f"特征工程 【目的】筛选有效特征，提升模型性能")

        # 获取特征列
        logger.print("\n提取特征列")
        logger.print("-" * 40)
        feature_cols = self.get_feature_columns(df)
        
        logger.print(f"  原始特征数: {len(feature_cols)}")

        ic_dict,ic_series = self.calculate_ic(df=df, feature_cols=feature_cols,
                                roll_win=roll_win, resampling_win=resampling_win)
        
        droped_features = ic_series[ic_series<ic_threshold].index.tolist()
        selected_features = ic_series[ic_series>=ic_threshold].index.tolist()

        df = df.drop(droped_features,axis=1)

        # 智能特征筛选
        selected_features = self.smart_feature_selection(
            df=df, feature_cols=selected_features, ic_dict=ic_dict,
            target_col=self.target_col,
            roll_win=roll_win,
            resampling_win=resampling_win,
            method=method
        )
        
        #logger.print(f"\n[特征筛选总结]")
        #logger.print(f"  原始特征数: {len(feature_cols)}")
        #logger.print(f"  筛选后特征数: {len(selected_features)}")
        #logger.print(f"  保留率: {len(selected_features)/len(feature_cols)*100:.1f}%")

        logger.panel(f"  原始特征数: {len(feature_cols)}"
                     f"  筛选后特征数: {len(selected_features)}"
                     f"  保留率: {len(selected_features)/len(feature_cols)*100:.1f}%","[特征筛选总结]")
        final_factors_ic = {f: ic_dict[f] for f in selected_features if f in ic_dict}
        logger.table(data=pd.DataFrame(list(final_factors_ic.items()), columns=["feature", "value"]).head(30), title="筛选后 Top 20 高IC因子")
        return selected_features, ic_dict