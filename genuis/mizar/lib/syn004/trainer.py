import re,pdb,time,os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List, Dict, Optional
from lib import logger

class Trainer(object):
    """
    支持两种训练方式：
    1. 单次训练：使用固定训练集和校验集
    2. 滚动训练：使用Walk-Forward Validation，每次用历史数据训练，预测未来数据
    """

    def __init__(self, params: Dict = None, train_params: Dict = None):
        """
        params: LightGBM模型参数（如果为None，使用默认参数）
        train_params: 训练参数（如果为None，使用默认参数）
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
        """
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 特征矩阵X、目标变量y、时间序列dates
        """


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
        
        

    def train_rolling(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                     selected_features: Optional[List[str]] = None,
                     n_splits: int = None) -> Tuple[List[lgb.Booster], pd.DataFrame]:
        """
        滚动训练（Walk-Forward Validation）
        【核心方法】这是时间序列预测的正确训练方式：
        - 每次用历史数据训练模型
        - 用训练好的模型预测未来数据
        - 避免数据泄露（不能用未来信息预测过去）

         Tuple[List[lgb.Booster], pd.DataFrame]: 每折的模型列表和验证结果
        """

        logger.rule("LightGBM模型训练（滚动训练 - Walk-Forward Validation）")

        logger.print("\n【目的】模拟真实交易场景，评估模型稳定性")
        logger.print("【方法】逐步向前验证，每次用历史数据训练，预测未来数据")
        logger.print("\n【关键】这是时间序列预测的正确训练方式！")
        logger.print("  ✓ 每次用历史数据训练")
        logger.print("  ✓ 用训练好的模型预测未来数据")
        logger.print("  ✗ 不能用未来信息预测过去（数据泄露）")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        models = []
        results = []
        
        logger.rule("执行{n_splits}折Walk-Forward验证:")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            logger.print(f"\n  Fold {fold}/{n_splits}:")
            logger.print(f"    训练集: {len(train_idx):,} 样本")
            logger.print(f"      时间: {dates[train_idx[0]]} 至 {dates[train_idx[-1]]}")
            logger.print(f"    测试集: {len(test_idx):,} 样本")
            logger.print(f"      时间: {dates[test_idx[0]]} 至 {dates[test_idx[-1]]}")

            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            logger.print(f"    训练模型中...")
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
                num_boost_round=300,#config.ROLLING_NUM_BOOST_ROUND,
                callbacks=[lgb.log_evaluation(period=0)]  # period=0表示不显示训练信息
            )

            models.append(model_fold)

            # 预测和评估
            logger.print(f"    预测和评估中...")
            y_pred = model_fold.predict(X_te)

            # 计算指标
            from scipy.stats import spearmanr
            ic = np.corrcoef(y_pred, y_te)[0, 1]
            rank_ic, _ = spearmanr(y_pred, y_te)
            rmse = np.sqrt(np.mean((y_pred - y_te)**2))
            direction_acc = np.mean(np.sign(y_pred) == np.sign(y_te))

            # 策略评估
            TRADING_DAYS_PER_YEAR = 252
            TRADING_MINUTES_PER_DAY = 240 # 日盘4小时
            PREDICTION_PERIOD_MINUTES = 15
            PERIODS_PER_DAY = TRADING_MINUTES_PER_DAY / PREDICTION_PERIOD_MINUTES
            PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR * PERIODS_PER_DAY
            strategy_returns = y_te * np.sign(y_pred)
            sharpe = (strategy_returns.mean() / strategy_returns.std() * 
                     np.sqrt(PERIODS_PER_YEAR)) if strategy_returns.std() > 0 else 0
            
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
            logger.print(f"    IC: {ic:.4f}, 方向准确率: {direction_acc:.2%}, Sharpe: {sharpe:.2f}")

        results_df = pd.DataFrame(results)

        logger.rule("Walk-Forward验证结果汇总:")
        logger.table(results_df, "训练结果")
        
        logger.print(f"\n  统计量:")
        logger.print(f"    平均IC: {results_df['IC'].mean():.4f} ± {results_df['IC'].std():.4f}")
        logger.print(f"    平均RankIC: {results_df['RankIC'].mean():.4f} ± {results_df['RankIC'].std():.4f}")
        logger.print(f"    平均方向准确率: {results_df['direction_acc'].mean():.2%} ± {results_df['direction_acc'].std():.2%}")
        logger.print(f"    平均Sharpe: {results_df['sharpe'].mean():.2f} ± {results_df['sharpe'].std():.2f}")
        
        # 稳定性评估
        ic_mean = results_df['IC'].mean()
        ic_std = results_df['IC'].std()
        ic_cv = ic_std / abs(ic_mean) if ic_mean != 0 else float('inf')
        
        logger.rule("稳定性评估:")
        logger.print(f"    IC变异系数: {ic_cv:.2f}")
        if ic_cv < 0.5:
            logger.print(f"    ✓ 模型稳定性良好")
        elif ic_cv < 1.0:
            logger.print(f"    ⚠ 模型稳定性一般")
        else:
            logger.print(f"    ✗ 模型稳定性较差")
        
        # 保存最后一个模型作为最终模型（用于后续预测）
        self.model = models[-1]
        self.best_iteration = 5
        
        return models, results_df
    


    def train_single(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None,
                    selected_features: Optional[List[str]] = None) -> lgb.Booster:
        content = f"""\n【说明】使用LightGBM的原因:\n  
                    1. 对中等规模数据（10万级）表现优异
                    2. 训练速度快
                    3. 对特征相关性不敏感
                    4. 内置正则化防止过拟合
                    5. 可解释性强（特征重要性）

        模型参数详解:\n"""
                        
        for key, value in self.params.items():
            content += f"    {key}: {value}\n"
        logger.panel(content=content, title="LightGBM模型训练（单次训练）")

        cleaned_feature_names = self.clean_feature_names(selected_features) if selected_features else None
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=cleaned_feature_names
        )

        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            logger.print(f"  转换验证数据为LightGBM格式...")
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                feature_name=cleaned_feature_names
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
        logger.print(f"  ✓ 数据集创建完成")
        
        logger.panel(f"  训练参数:"
                     f"    最大迭代次数: {self.train_params['num_boost_round']}"
                     f"    早停轮数: {self.train_params['early_stopping_rounds']}"
                     f"    显示间隔: {self.train_params.get('verbose_eval', 50)}",
                     "开始训练模型")

        logger.print(f"\n  训练中...")
        logger.print(f"  " + "-" * 40)
        
        train_start_time = time.time()

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

        content = f"    训练用时: {training_time:.2f} 秒"
        content += f"    最佳迭代: {self.model.best_iteration}"
        
        # 动态获取评估指标名称
        if hasattr(self.model, 'best_score'):
            # 获取训练集上的评估指标
            if 'train' in self.model.best_score:
                train_scores = self.model.best_score['train']
                if train_scores:
                    metric_name = list(train_scores.keys())[0]  # 获取第一个指标名称
                    metric_value = train_scores[metric_name]
                    content += f"    训练集最佳{metric_name.upper()}: {metric_value:.6f}"

            # 获取验证集上的评估指标
            if 'valid' in self.model.best_score:
                valid_scores = self.model.best_score['valid']
                if valid_scores:
                    metric_name = list(valid_scores.keys())[0]  # 获取第一个指标名称
                    metric_value = valid_scores[metric_name]
                    content += f"    验证集最佳{metric_name.upper()}: {metric_value:.6f}"

        logger.panel(content=content, title="✓ 训练完成！")
        return self.model

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
    
    def save(self, output:str, model: Optional[lgb.Booster]):
        model_path = os.path.join(output, 'lgb_model.txt')
        model.save_model(model_path)
