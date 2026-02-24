import re,pdb,time,os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List, Dict, Optional
from lib import logger
from lib.cux001 import FactorEvaluate1

class Trainer(object):
    def __init__(self, params: Dict = None, train_params: Dict = None):
        """
        params: 模型参数（如果为None，使用默认参数）
        train_params: 训练参数（如果为None，则代表没有训练参数）
        """
        self.params = params
        self.train_params = train_params
        self.model = None
        self.best_iteration = None
        self.feature_name_mapping = {}  # 存储原始特征名到清理后特征名的映射

    def clean_feature_names(self, feature_names: List[str]) -> List[str]:
        """
        清理特征名称，移除LightGBM不支持的特殊字符
        
        """
        cleaned_names = []
        seen_names = {}  # 用于跟踪已使用的清理后名称，避免重复
        
        for idx, name in enumerate(feature_names):
            # 替换所有特殊字符为下划线
            # 保留字母、数字、下划线、点号
            cleaned = re.sub(r'[^a-zA-Z0-9_.]', '_', str(name))
            # 移除连续的下划线
            cleaned = re.sub(r'_+', '_', cleaned)
            # 移除开头和结尾的下划线
            cleaned = cleaned.strip('_')
            # 如果清理后为空，使用默认名称
            if not cleaned:
                cleaned = f'feature_{idx}'
            
            # 处理重复的清理后名称
            original_cleaned = cleaned
            counter = 0
            while cleaned in seen_names:
                counter += 1
                cleaned = f'{original_cleaned}_{counter}'
            
            seen_names[cleaned] = True
            cleaned_names.append(cleaned)
            # 保存映射关系
            self.feature_name_mapping[cleaned] = name
        
        return cleaned_names

    def prepare_data(self, df: pd.DataFrame, 
                    selected_features: List[str],
                    taget_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        X = df[selected_features].values
        y = df[taget_col].values
        dates = df['trade_time'].values

        logger.panel(f"  特征矩阵 X: {X.shape} (样本数 × 特征数) \n"
                     f"  目标变量 y: {y.shape}"
                     f"  时间序列: {len(dates)}\n"
                     f"\n  数据类型:\n"
                     f"    X: {X.dtype}\n"
                     f"    y: {y.dtype}\n"
                     f"\n  数据范围:\n"
                     f"    X最小值: {X.min():.6f}\n"
                     f"    X最大值: {X.max():.6f}\n"
                     f"    y最小值: {y.min():.6f}\n"
                     f"    y最大值: {y.max():.6f}",
                     "提取特征矩阵和目标变量")
        return X, y, dates

    def split_data(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                   train_ratio: float = 0.7) -> Tuple:
        """
        按时间顺序划分训练集和校验集
        
        【重要】时间序列预测必须按时间顺序划分，不能随机划分！
        
        参数:
            X: 特征矩阵
            y: 目标变量
            dates: 时间序列
            train_ratio: 训练集比例
        
        返回:
            Tuple: (X_train, X_val, y_train, y_val, dates_train, dates_val)
        """

        logger.panel(
            f"  ✓ 正确: 前70%训练，后30%校验\n"
            "  ✗ 错误: 随机划分（会导致用未来预测过去\n",
            title="时间序列预测必须按时间顺序划分数据"
        )
        split_idx = int(len(X) * train_ratio)

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        dates_train, dates_val = dates[:split_idx], dates[split_idx:]

        logger.panel(
            f"  训练/校验比例: {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%\n"
            f"  划分点索引: {split_idx}\n"
            f"\n  [训练集]\n"
            f"    样本数: {len(X_train):,}\n"
            f"    时间范围: {dates_train[0]} 至 {dates_train[-1]}\n"
            f"    时间跨度: {(pd.Timestamp(dates_train[-1]) - pd.Timestamp(dates_train[0])).days} 天\n"
            f"    目标变量统计:\n"
            f"      均值: {y_train.mean():.6f}\n"
            f"      标准差: {y_train.std():.6f}\n"
            f"      正收益比例: {(y_train > 0).mean()*100:.2f}%\n"
            f"\n  [校验集]\n"
            f"    样本数: {len(X_val):,}\n"
            f"    时间范围: {dates_val[0]} 至 {dates_val[-1]}\n"
            f"    时间跨度: {(pd.Timestamp(dates_val[-1]) - pd.Timestamp(dates_val[0])).days} 天\n"
            f"    目标变量统计:\n"
            f"      均值: {y_val.mean():.6f}\n"
            f"      标准差: {y_val.std():.6f}\n"
            f"      正收益比例: {(y_val > 0).mean()*100:.2f}%\n", title="数据集信息"
        )


        # 检查训练集和测试集分布差异
        mean_diff = abs(y_train.mean() - y_val.mean())
        std_ratio = y_val.std() / y_train.std()
        
        content = f"    均值差异: {mean_diff:.6f}\n"
        content += f"    标准差比: {std_ratio:.2f}\n"

        if std_ratio > 1.5 or std_ratio < 0.67:
            content+= f"    ⚠️  警告: 测试集波动性与训练集差异较大\n"
        else:
            content+= f"    ✓ 训练集和测试集分布相对一致\n"

        logger.panel(
            content=content,title="[分布一致性检查]"
        )
        
        return X_train, X_val, y_train, y_val, dates_train, dates_val

    
    def train_single(self, model_class, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None,
                    selected_features: Optional[List[str]] = None):
        content = f"""
        模型参数详解:\n"""

        for key, value in self.params.items():
            content += f"    {key}: {value}\n"
        logger.panel(content=content, title="模型训练（单次训练）")

        cleaned_feature_names = self.clean_feature_names(selected_features) if selected_features else None
        y_train_scaled = y_train * 1000 
        self.model = model_class(**self.params) if len(self.params) > 0 else model_class()
        self.model.fit(X_train, y_train_scaled)

        y_pred_val = self.model.predict(X_val)
        
        val_mae = np.mean(np.abs(y_pred_val - y_val))

        logger.panel(content=f"Intercept (截距): {self.model.intercept_}\n"
            f"Coefficients (系数): {self.model.coef_}\n"
            f"Sample Prediction (前5个预测值): {self.model.predict(X_train[:5])}\n"
            f"Target Y Mean (目标均值): {y_train.mean()}\n"
            f"本验证集 MAE: {val_mae:.6f}\n", 
            title="✓ 训练完成！")
        return self.model

    def predict(self, X: np.ndarray, model = None) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X: 特征矩阵
            model: 模型（如果为None，使用self.model）
        
        返回:
            np.ndarray: 预测值
        """
        if model is None:
            if self.model is None:
                raise ValueError("模型未训练，请先调用train_single或train_rolling")
            model = self.model
        return model.predict(X)

    
    def predict_all(self, X: np.ndarray, dates_val:np.ndarray, code:str, period:int, roll_win:int,
                    data:pd.DataFrame, expression:str, outdirs:str, title:str, model = None):
        #pdb.set_trace()
        y_pred = self.predict(X, model)
        val_factors = pd.Series(y_pred, index=pd.MultiIndex.from_arrays(
            [dates_val, [code] * len(dates_val)],names=['trade_time', 'code']    # 为每一层索引命名
        ), name='transformed')

        val_factors = val_factors.reset_index().merge(data[['trade_time','code','nxt1_ret_{0}h'.format(period)]], 
                on=['trade_time','code'])

        evaluate1 = FactorEvaluate1(factor_data=val_factors,
                                factor_name='transformed',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=roll_win,
                                fee=0.000,
                                scale_method='raw',
                                expression=expression,
                                resampling_win=roll_win)
        state1 = evaluate1.run()
        #pdb.set_trace()
        logger.table(pd.DataFrame([state1]), title="{0} 绩效".format(title))
        evaluate1.plot_results()
        evaluate1.save_results(os.path.join(outdirs, expression))
