"""
模型训练模块

负责模型训练工作，支持滚动训练（Walk-Forward Validation）方式。
这是时间序列预测的关键：必须使用滚动训练，避免数据泄露。
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List, Dict, Optional
import time
import re
from pathlib import Path
try:
    from . import config
except ImportError:
    import config


class ModelTrainer:
    """
    模型训练器类
    
    支持两种训练方式：
    1. 单次训练：使用固定训练集和测试集
    2. 滚动训练：使用Walk-Forward Validation，每次用历史数据训练，预测未来数据
    """
    
    def __init__(self, params: Dict = None, train_params: Dict = None):
        """
        初始化模型训练器
        
        参数:
            params: LightGBM模型参数（如果为None，使用默认参数）
            train_params: 训练参数（如果为None，使用默认参数）
        """
        self.params = params if params is not None else config.LGB_PARAMS.copy()
        self.train_params = train_params if train_params is not None else config.TRAIN_PARAMS.copy()
        self.model = None
        self.best_iteration = None
        self.feature_name_mapping = {}  # 存储原始特征名到清理后特征名的映射
    
    def clean_feature_names(self, feature_names: List[str]) -> List[str]:
        """
        清理特征名称，移除LightGBM不支持的特殊字符
        
        LightGBM不支持的特征名称字符包括：
        - 空格
        - 特殊JSON字符: : , [ ] { } " ' 等
        
        参数:
            feature_names: 原始特征名称列表
        
        返回:
            List[str]: 清理后的特征名称列表
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
                    selected_features: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        参数:
            df: 数据框
            selected_features: 选择的特征列表
        
        返回:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 特征矩阵X、目标变量y、时间序列dates
        """
        print("\n[5.1] 提取特征矩阵和目标变量")
        print("-" * 40)
        
        X = df[selected_features].values
        y = df[config.TARGET_COL].values
        dates = df['trade_time'].values
        
        print(f"  特征矩阵 X: {X.shape} (样本数 × 特征数)")
        print(f"  目标变量 y: {y.shape}")
        print(f"  时间序列: {len(dates)}")
        
        print(f"\n  数据类型:")
        print(f"    X: {X.dtype}")
        print(f"    y: {y.dtype}")
        
        print(f"\n  数据范围:")
        print(f"    X最小值: {X.min():.6f}")
        print(f"    X最大值: {X.max():.6f}")
        print(f"    y最小值: {y.min():.6f}")
        print(f"    y最大值: {y.max():.6f}")
        
        return X, y, dates
    
    def split_data(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                   train_ratio: float = 0.7) -> Tuple:
        """
        按时间顺序划分训练集和测试集
        
        【重要】时间序列预测必须按时间顺序划分，不能随机划分！
        
        参数:
            X: 特征矩阵
            y: 目标变量
            dates: 时间序列
            train_ratio: 训练集比例
        
        返回:
            Tuple: (X_train, X_test, y_train, y_test, dates_train, dates_test)
        """
        print("\n[5.2] 时间序列划分")
        print("-" * 40)
        
        print("\n【关键】时间序列预测必须按时间顺序划分数据！")
        print("  ✓ 正确: 前70%训练，后30%测试")
        print("  ✗ 错误: 随机划分（会导致用未来预测过去）")
        
        split_idx = int(len(X) * train_ratio)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_train, dates_test = dates[:split_idx], dates[split_idx:]
        
        print(f"  训练/测试比例: {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%")
        print(f"  划分点索引: {split_idx}")
        
        print(f"\n  [训练集]")
        print(f"    样本数: {len(X_train):,}")
        print(f"    时间范围: {dates_train[0]} 至 {dates_train[-1]}")
        print(f"    时间跨度: {(pd.Timestamp(dates_train[-1]) - pd.Timestamp(dates_train[0])).days} 天")
        print(f"    目标变量统计:")
        print(f"      均值: {y_train.mean():.6f}")
        print(f"      标准差: {y_train.std():.6f}")
        print(f"      正收益比例: {(y_train > 0).mean()*100:.2f}%")
        
        print(f"\n  [测试集]")
        print(f"    样本数: {len(X_test):,}")
        print(f"    时间范围: {dates_test[0]} 至 {dates_test[-1]}")
        print(f"    时间跨度: {(pd.Timestamp(dates_test[-1]) - pd.Timestamp(dates_test[0])).days} 天")
        print(f"    目标变量统计:")
        print(f"      均值: {y_test.mean():.6f}")
        print(f"      标准差: {y_test.std():.6f}")
        print(f"      正收益比例: {(y_test > 0).mean()*100:.2f}%")
        
        # 检查训练集和测试集分布差异
        print(f"\n  [分布一致性检查]")
        mean_diff = abs(y_train.mean() - y_test.mean())
        std_ratio = y_test.std() / y_train.std()
        print(f"    均值差异: {mean_diff:.6f}")
        print(f"    标准差比: {std_ratio:.2f}")
        if std_ratio > 1.5 or std_ratio < 0.67:
            print(f"    ⚠️  警告: 测试集波动性与训练集差异较大")
        else:
            print(f"    ✓ 训练集和测试集分布相对一致")
        
        return X_train, X_test, y_train, y_test, dates_train, dates_test
    
    def train_single(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None,
                    selected_features: Optional[List[str]] = None) -> lgb.Booster:
        """
        单次训练模型（使用固定训练集和验证集）
        
        参数:
            X_train: 训练集特征
            y_train: 训练集目标变量
            X_val: 验证集特征（可选）
            y_val: 验证集目标变量（可选）
            selected_features: 特征名称列表（可选，用于特征重要性分析）
        
        返回:
            lgb.Booster: 训练好的模型
        """
        print("\n" + "=" * 80)
        print("第6步：LightGBM模型训练（单次训练）")
        print("=" * 80)
        
        print("\n【说明】使用LightGBM的原因:")
        print("  1. 对中等规模数据（10万级）表现优异")
        print("  2. 训练速度快")
        print("  3. 对特征相关性不敏感")
        print("  4. 内置正则化防止过拟合")
        print("  5. 可解释性强（特征重要性）")
        
        print("\n[6.1] 设置模型参数")
        print("-" * 40)
        print(f"  模型参数详解:")
        for key, value in self.params.items():
            print(f"    {key}: {value}")
        
        print("\n[6.2] 创建LightGBM数据集")
        print("-" * 40)
        
        print(f"  转换训练数据为LightGBM格式...")
        # 清理特征名称
        cleaned_feature_names = self.clean_feature_names(selected_features) if selected_features else None
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=cleaned_feature_names
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            print(f"  转换验证数据为LightGBM格式...")
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                feature_name=cleaned_feature_names
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        print(f"  ✓ 数据集创建完成")
        
        print("\n[6.3] 开始训练模型")
        print("-" * 40)
        
        print(f"  训练参数:")
        print(f"    最大迭代次数: {self.train_params['num_boost_round']}")
        print(f"    早停轮数: {self.train_params['early_stopping_rounds']}")
        print(f"    显示间隔: {self.train_params.get('verbose_eval', 50)}")
        
        print(f"\n  训练中...")
        print(f"  " + "-" * 40)
        
        train_start_time = time.time()
        
        # 训练模型
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.train_params['num_boost_round'],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.train_params['early_stopping_rounds'],
                    verbose=True
                ),
                lgb.log_evaluation(
                    period=self.train_params.get('verbose_eval', 50)
                )
            ]
        )
        
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        
        self.best_iteration = self.model.best_iteration
        
        print(f"  " + "-" * 40)
        print(f"  ✓ 训练完成！")
        print(f"    训练用时: {training_time:.2f} 秒")
        print(f"    最佳迭代: {self.model.best_iteration}")
        if hasattr(self.model, 'best_score'):
            print(f"    训练集最佳RMSE: {self.model.best_score['train']['rmse']:.6f}")
            if 'valid' in self.model.best_score:
                print(f"    验证集最佳RMSE: {self.model.best_score['valid']['rmse']:.6f}")
        
        return self.model
    
    def train_rolling(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                     selected_features: Optional[List[str]] = None,
                     n_splits: int = None) -> Tuple[List[lgb.Booster], pd.DataFrame]:
        """
        滚动训练（Walk-Forward Validation）
        
        【核心方法】这是时间序列预测的正确训练方式：
        - 每次用历史数据训练模型
        - 用训练好的模型预测未来数据
        - 避免数据泄露（不能用未来信息预测过去）
        
        参数:
            X: 完整特征矩阵
            y: 完整目标变量
            dates: 时间序列
            selected_features: 特征名称列表（可选）
            n_splits: 交叉验证折数
        
        返回:
            Tuple[List[lgb.Booster], pd.DataFrame]: 每折的模型列表和验证结果
        """
        if n_splits is None:
            n_splits = config.N_SPLITS
        
        print("\n" + "=" * 80)
        print("第6步：LightGBM模型训练（滚动训练 - Walk-Forward Validation）")
        print("=" * 80)
        
        print("\n【目的】模拟真实交易场景，评估模型稳定性")
        print("【方法】逐步向前验证，每次用历史数据训练，预测未来数据")
        print("\n【关键】这是时间序列预测的正确训练方式！")
        print("  ✓ 每次用历史数据训练")
        print("  ✓ 用训练好的模型预测未来数据")
        print("  ✗ 不能用未来信息预测过去（数据泄露）")
        
        # 初始化TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        models = []
        results = []
        
        print(f"\n  执行{n_splits}折Walk-Forward验证:")
        print(f"  " + "=" * 60)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            print(f"\n  Fold {fold}/{n_splits}:")
            print(f"    训练集: {len(train_idx):,} 样本")
            print(f"      时间: {dates[train_idx[0]]} 至 {dates[train_idx[-1]]}")
            print(f"    测试集: {len(test_idx):,} 样本")
            print(f"      时间: {dates[test_idx[0]]} 至 {dates[test_idx[-1]]}")
            
            # 划分数据
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            # 【关键】每折重新训练模型
            print(f"    训练模型中...")
            # 清理特征名称
            cleaned_feature_names = self.clean_feature_names(selected_features) if selected_features else None
            train_set = lgb.Dataset(
                X_tr,
                label=y_tr,
                feature_name=cleaned_feature_names
            )
            
            # 使用较小的迭代次数，因为每折数据量较小
            model_fold = lgb.train(
                self.params,
                train_set,
                num_boost_round=config.ROLLING_NUM_BOOST_ROUND,
                callbacks=[lgb.log_evaluation(period=0)]  # period=0表示不显示训练信息
            )
            
            models.append(model_fold)
            
            # 预测和评估
            print(f"    预测和评估中...")
            y_pred = model_fold.predict(X_te)
            
            # 计算指标
            from scipy.stats import spearmanr
            ic = np.corrcoef(y_pred, y_te)[0, 1]
            rank_ic, _ = spearmanr(y_pred, y_te)
            rmse = np.sqrt(np.mean((y_pred - y_te)**2))
            direction_acc = np.mean(np.sign(y_pred) == np.sign(y_te))
            
            # 策略评估
            strategy_returns = y_te * np.sign(y_pred)
            sharpe = (strategy_returns.mean() / strategy_returns.std() * 
                     np.sqrt(config.PERIODS_PER_YEAR)) if strategy_returns.std() > 0 else 0
            
            results.append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_start': dates[train_idx[0]],
                'train_end': dates[train_idx[-1]],
                'test_start': dates[test_idx[0]],
                'test_end': dates[test_idx[-1]],
                'IC': ic,
                'RankIC': rank_ic,
                'RMSE': rmse,
                'direction_acc': direction_acc,
                'sharpe': sharpe,
                'cum_return': strategy_returns.sum()
            })
            
            print(f"    IC: {ic:.4f}, 方向准确率: {direction_acc:.2%}, Sharpe: {sharpe:.2f}")
        
        results_df = pd.DataFrame(results)
        
        print(f"\n  " + "=" * 60)
        print(f"  Walk-Forward验证结果汇总:")
        print(f"  " + "=" * 60)
        print(results_df.to_string(index=False))
        
        print(f"\n  统计量:")
        print(f"    平均IC: {results_df['IC'].mean():.4f} ± {results_df['IC'].std():.4f}")
        print(f"    平均RankIC: {results_df['RankIC'].mean():.4f} ± {results_df['RankIC'].std():.4f}")
        print(f"    平均方向准确率: {results_df['direction_acc'].mean():.2%} ± {results_df['direction_acc'].std():.2%}")
        print(f"    平均Sharpe: {results_df['sharpe'].mean():.2f} ± {results_df['sharpe'].std():.2f}")
        
        # 稳定性评估
        ic_mean = results_df['IC'].mean()
        ic_std = results_df['IC'].std()
        ic_cv = ic_std / abs(ic_mean) if ic_mean != 0 else float('inf')
        
        print(f"\n  稳定性评估:")
        print(f"    IC变异系数: {ic_cv:.2f}")
        if ic_cv < 0.5:
            print(f"    ✓ 模型稳定性良好")
        elif ic_cv < 1.0:
            print(f"    ⚠ 模型稳定性一般")
        else:
            print(f"    ✗ 模型稳定性较差")
        
        # 保存最后一个模型作为最终模型（用于后续预测）
        self.model = models[-1]
        self.best_iteration = config.ROLLING_NUM_BOOST_ROUND
        
        return models, results_df
    
    def predict(self, X: np.ndarray, model: Optional[lgb.Booster] = None) -> np.ndarray:
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
        
        num_iteration = self.best_iteration if self.best_iteration else None
        return model.predict(X, num_iteration=num_iteration)

    def load_model(self, model_path: str, best_iteration: Optional[int] = None) -> lgb.Booster:
        """
        从磁盘加载已经训练好的模型
        
        参数:
            model_path: 保存的LightGBM模型文件路径
            best_iteration: 可选，最佳迭代次数。若未提供，将尝试从模型中读取
        
        返回:
            lgb.Booster: 加载的模型
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print("\n[10.1] 加载已保存的LightGBM模型")
        print("-" * 40)
        print(f"  模型路径: {model_path}")
        
        booster = lgb.Booster(model_file=model_path)
        
        self.model = booster
        if best_iteration is not None:
            self.best_iteration = best_iteration
        else:
            self.best_iteration = booster.best_iteration or booster.current_iteration()
        
        print(f"  ✓ 模型加载完成，最佳迭代: {self.best_iteration}")
        return booster

