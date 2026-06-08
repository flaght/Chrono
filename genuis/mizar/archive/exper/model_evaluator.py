"""
模型评估模块

负责模型性能评估，包括：
1. 基础回归指标（RMSE、MAE）
2. 相关性指标（IC、RankIC）
3. 方向准确率
4. 策略回测指标（Sharpe Ratio、最大回撤等）
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from typing import Tuple, Dict
try:
    from . import config
except ImportError:
    import config


class ModelEvaluator:
    """
    模型评估器类
    
    提供完整的模型性能评估功能。
    """
    
    def __init__(self):
        """初始化模型评估器"""
        pass  # 无需初始化参数
    
    def calculate_regression_metrics(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算基础回归指标
        
        参数:
            y_true: 真实值
            y_pred: 预测值
        
        返回:
            Dict[str, float]: 包含RMSE和MAE的字典
        """
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        mae = np.mean(np.abs(y_pred - y_true))
        
        return {
            'rmse': rmse,
            'mae': mae
        }
    
    def calculate_correlation_metrics(self, y_true: np.ndarray,
                                     y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算相关性指标
        
        参数:
            y_true: 真实值
            y_pred: 预测值
        
        返回:
            Dict[str, float]: 包含IC和RankIC的字典
        """
        # IC (Pearson相关系数)
        ic = np.corrcoef(y_pred, y_true)[0, 1]
        
        # RankIC (Spearman秩相关)
        rank_ic, _ = spearmanr(y_pred, y_true)
        
        return {
            'ic': ic,
            'rank_ic': rank_ic
        }
    
    def calculate_direction_accuracy(self, y_true: np.ndarray,
                                    y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算方向准确率
        
        参数:
            y_true: 真实值
            y_pred: 预测值
        
        返回:
            Tuple[float, np.ndarray]: (方向准确率, 混淆矩阵)
        """
        pred_direction = np.sign(y_pred)
        true_direction = np.sign(y_true)
        
        direction_acc = np.mean(pred_direction == true_direction)
        
        # 混淆矩阵
        cm = confusion_matrix(true_direction, pred_direction, labels=[-1, 1])
        
        return direction_acc, cm
    
    def calculate_strategy_returns(self, y_true: np.ndarray,
                                  y_pred: np.ndarray) -> np.ndarray:
        """
        计算策略收益
        
        策略逻辑：
        - 预测收益 > 0 → 做多（收益 = 实际收益）
        - 预测收益 < 0 → 做空（收益 = -实际收益）
        - 预测收益 = 0 → 不交易（收益 = 0）
        
        参数:
            y_true: 真实收益
            y_pred: 预测收益
        
        返回:
            np.ndarray: 策略收益序列
        """
        return y_true * np.sign(y_pred)
    
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        计算年化Sharpe Ratio
        
        参数:
            returns: 收益序列
        
        返回:
            float: 年化Sharpe Ratio
        """
        if returns.std() == 0:
            return 0.0
        
        sharpe = (returns.mean() / returns.std() * 
                 np.sqrt(config.PERIODS_PER_YEAR))
        return sharpe
    
    def calculate_max_drawdown(self, returns: np.ndarray) -> Tuple[float, np.ndarray, int]:
        """
        计算最大回撤
        
        参数:
            returns: 收益序列
        
        返回:
            Tuple[float, np.ndarray, int]: (最大回撤, 回撤序列, 最大回撤位置索引)
        """
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        return max_dd, drawdown, max_dd_idx
    
    def calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """
        计算卡玛比率（Calmar Ratio）
        
        卡玛比率 = 累计收益率 / 最大回撤的绝对值
        
        参照 cux001.py 的实现方式
        
        参数:
            returns: 收益序列
        
        返回:
            float: 卡玛比率
        """
        # 计算累计净值
        nav = (1 + returns).cumprod()
        total_ret = nav[-1] - 1 if len(nav) > 0 else 0
        
        # 计算最大回撤
        max_dd, _, _ = self.calculate_max_drawdown(returns)
        
        # 卡玛比率 = 累计收益率 / 最大回撤的绝对值
        if max_dd != 0:
            calmar = total_ret / abs(max_dd)
        else:
            calmar = np.nan
        
        return calmar
    
    def calculate_quantile_analysis(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   n_quantiles: int = 5) -> pd.DataFrame:
        """
        分位数分析：验证预测值大小与实际收益的单调关系
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            n_quantiles: 分位数数量
        
        返回:
            pd.DataFrame: 分位数统计结果
        """
        df = pd.DataFrame({
            'pred': y_pred,
            'actual': y_true
        })
        
        # 分成n_quantiles个分位数
        labels = [f'Q{i+1}' for i in range(n_quantiles)]
        if n_quantiles == 5:
            labels = ['Q1(最看空)', 'Q2', 'Q3', 'Q4', 'Q5(最看多)']
        
        df['pred_quantile'] = pd.qcut(
            df['pred'],
            q=n_quantiles,
            labels=labels,
            duplicates='drop'
        )
        
        quantile_stats = df.groupby('pred_quantile')['actual'].agg(['mean', 'std', 'count'])
        
        return quantile_stats
    
    def evaluate(self, y_train: np.ndarray, y_train_pred: np.ndarray,
                y_test: np.ndarray, y_test_pred: np.ndarray,
                dates_test: np.ndarray = None) -> Dict:
        """
        执行完整的模型评估
        
        参数:
            y_train: 训练集真实值
            y_train_pred: 训练集预测值
            y_test: 测试集真实值
            y_test_pred: 测试集预测值
            dates_test: 测试集时间序列（可选，用于显示最大回撤发生时间）
        
        返回:
            Dict: 包含所有评估指标的字典
        """
        print("\n" + "=" * 80)
        print("第8步：模型评估（时序单品预测）")
        print("=" * 80)
        
        print("\n【评估说明】")
        print("  时序单品预测的核心指标：")
        print("    1. ⭐ 方向准确率 - 预测涨跌方向是否正确")
        print("    2. ⭐ Sharpe Ratio - 策略风险调整后收益")
        print("    3. ⭐ 策略累计收益 - 模拟交易的总收益")
        print("    4. IC/RankIC - 预测值与实际值的相关性")
        print("    5. 最大回撤 - 策略的风险度量")
        
        # 1. 基础回归指标
        print("\n[8.1] 基础回归指标")
        print("-" * 40)
        
        train_metrics = self.calculate_regression_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_regression_metrics(y_test, y_test_pred)
        
        print(f"  RMSE (均方根误差):")
        print(f"    训练集: {train_metrics['rmse']:.6f}")
        print(f"    测试集: {test_metrics['rmse']:.6f}")
        overfit_ratio = (test_metrics['rmse']/train_metrics['rmse'] - 1) * 100
        print(f"    过拟合比例: {overfit_ratio:.2f}%")
        
        print(f"\n  MAE (平均绝对误差):")
        print(f"    训练集: {train_metrics['mae']:.6f}")
        print(f"    测试集: {test_metrics['mae']:.6f}")
        
        # 2. 相关性指标
        print("\n[8.2] 相关性指标")
        print("-" * 40)
        
        train_corr = self.calculate_correlation_metrics(y_train, y_train_pred)
        test_corr = self.calculate_correlation_metrics(y_test, y_test_pred)
        
        print(f"  IC (Pearson相关系数):")
        print(f"    训练集: {train_corr['ic']:.4f}")
        print(f"    测试集: {test_corr['ic']:.4f}")
        
        print(f"\n  RankIC (Spearman秩相关):")
        print(f"    训练集: {train_corr['rank_ic']:.4f}")
        print(f"    测试集: {test_corr['rank_ic']:.4f}")
        
        print(f"\n  【IC解释】")
        print(f"    IC>0.05: 优秀（可实盘）")
        print(f"    IC>0.03: 良好")
        print(f"    IC>0.01: 一般")
        print(f"    IC<0.01: 较差")
        
        # 3. 方向准确率
        print("\n[8.3] 方向预测能力 ⭐ (最关键)")
        print("-" * 40)
        
        train_dir_acc, train_cm = self.calculate_direction_accuracy(y_train, y_train_pred)
        test_dir_acc, test_cm = self.calculate_direction_accuracy(y_test, y_test_pred)
        
        print(f"  方向准确率:")
        print(f"    训练集: {train_dir_acc*100:.2f}%")
        print(f"    测试集: {test_dir_acc*100:.2f}%")
        
        # 评级
        if test_dir_acc > 0.55:
            rating = "✓✓ 优秀"
            comment = "可实盘"
        elif test_dir_acc > 0.52:
            rating = "✓  良好"
            comment = "需谨慎"
        elif test_dir_acc > 0.50:
            rating = "⚠  一般"
            comment = "需优化"
        else:
            rating = "✗  不可用"
            comment = "重新设计"
        
        print(f"\n  【评级】")
        print(f"    {rating} ({comment})")
        
        # 混淆矩阵
        print("\n[8.4] 方向预测混淆矩阵")
        print("-" * 40)
        
        print(f"  混淆矩阵:")
        print(f"                预测下跌  预测上涨")
        print(f"    实际下跌    {test_cm[0,0]:6d}    {test_cm[0,1]:6d}")
        print(f"    实际上涨    {test_cm[1,0]:6d}    {test_cm[1,1]:6d}")
        
        print(f"\n  详细分析:")
        print(f"    真阴性(TN): {test_cm[0,0]} - 正确预测下跌")
        print(f"    假阳性(FP): {test_cm[0,1]} - 错误预测上涨")
        print(f"    假阴性(FN): {test_cm[1,0]} - 错误预测下跌")
        print(f"    真阳性(TP): {test_cm[1,1]} - 正确预测上涨")
        
        # 做多指标
        precision_up = (test_cm[1,1] / (test_cm[0,1] + test_cm[1,1]) 
                       if (test_cm[0,1] + test_cm[1,1]) > 0 else 0)
        recall_up = (test_cm[1,1] / (test_cm[1,0] + test_cm[1,1]) 
                    if (test_cm[1,0] + test_cm[1,1]) > 0 else 0)
        
        # 做空指标
        precision_down = (test_cm[0,0] / (test_cm[0,0] + test_cm[1,0]) 
                         if (test_cm[0,0] + test_cm[1,0]) > 0 else 0)
        recall_down = (test_cm[0,0] / (test_cm[0,0] + test_cm[0,1]) 
                      if (test_cm[0,0] + test_cm[0,1]) > 0 else 0)
        
        print(f"\n  性能指标:")
        print(f"    【做多】")
        print(f"      做多精度: {precision_up*100:.2f}% (预测上涨时，实际上涨的概率)")
        print(f"      做多召回: {recall_up*100:.2f}% (实际上涨时，成功预测的概率)")
        print(f"    【做空】")
        print(f"      做空精度: {precision_down*100:.2f}% (预测下跌时，实际下跌的概率)")
        print(f"      做空召回: {recall_down*100:.2f}% (实际下跌时，成功预测的概率)")
        
        # 4. 策略回测
        print("\n[8.5] 简单方向性策略回测 ⭐")
        print("-" * 40)
        
        print(f"  【策略逻辑】")
        print(f"    预测收益 > 0 → 做多（收益 = 实际收益）")
        print(f"    预测收益 < 0 → 做空（收益 = -实际收益）")
        print(f"    预测收益 = 0 → 不交易（收益 = 0）")
        
        train_strategy_returns = self.calculate_strategy_returns(y_train, y_train_pred)
        test_strategy_returns = self.calculate_strategy_returns(y_test, y_test_pred)
        
        # 分离做多和做空收益
        train_pred_direction = np.sign(y_train_pred)
        test_pred_direction = np.sign(y_test_pred)
        
        train_long_returns = train_strategy_returns[train_pred_direction > 0]
        train_short_returns = train_strategy_returns[train_pred_direction < 0]
        test_long_returns = test_strategy_returns[test_pred_direction > 0]
        test_short_returns = test_strategy_returns[test_pred_direction < 0]
        
        print(f"\n  [训练集策略表现]")
        print(f"    累计收益: {train_strategy_returns.sum():.6f}")
        print(f"    平均收益: {train_strategy_returns.mean():.6f}")
        print(f"    收益标准差: {train_strategy_returns.std():.6f}")
        print(f"    胜率: {(train_strategy_returns > 0).mean()*100:.2f}%")
        
        win_loss_ratio_train = (
            train_strategy_returns[train_strategy_returns > 0].mean() /
            abs(train_strategy_returns[train_strategy_returns < 0].mean())
            if len(train_strategy_returns[train_strategy_returns < 0]) > 0 else 0
        )
        print(f"    盈亏比: {win_loss_ratio_train:.2f}")
        
        print(f"\n    [做多统计]")
        if len(train_long_returns) > 0:
            print(f"      做多次数: {len(train_long_returns)}")
            print(f"      做多累计收益: {train_long_returns.sum():.6f}")
            print(f"      做多平均收益: {train_long_returns.mean():.6f}")
            print(f"      做多胜率: {(train_long_returns > 0).mean()*100:.2f}%")
        else:
            print(f"      做多次数: 0")
        
        print(f"\n    [做空统计]")
        if len(train_short_returns) > 0:
            print(f"      做空次数: {len(train_short_returns)}")
            print(f"      做空累计收益: {train_short_returns.sum():.6f}")
            print(f"      做空平均收益: {train_short_returns.mean():.6f}")
            print(f"      做空胜率: {(train_short_returns > 0).mean()*100:.2f}%")
        else:
            print(f"      做空次数: 0")
        
        print(f"\n  [测试集策略表现]")
        print(f"    累计收益: {test_strategy_returns.sum():.6f}")
        print(f"    平均收益: {test_strategy_returns.mean():.6f}")
        print(f"    收益标准差: {test_strategy_returns.std():.6f}")
        print(f"    胜率: {(test_strategy_returns > 0).mean()*100:.2f}%")
        
        win_loss_ratio_test = (
            test_strategy_returns[test_strategy_returns > 0].mean() /
            abs(test_strategy_returns[test_strategy_returns < 0].mean())
            if len(test_strategy_returns[test_strategy_returns < 0]) > 0 else 0
        )
        print(f"    盈亏比: {win_loss_ratio_test:.2f}")
        
        print(f"\n    [做多统计]")
        if len(test_long_returns) > 0:
            print(f"      做多次数: {len(test_long_returns)}")
            print(f"      做多累计收益: {test_long_returns.sum():.6f}")
            print(f"      做多平均收益: {test_long_returns.mean():.6f}")
            print(f"      做多胜率: {(test_long_returns > 0).mean()*100:.2f}%")
        else:
            print(f"      做多次数: 0")
        
        print(f"\n    [做空统计]")
        if len(test_short_returns) > 0:
            print(f"      做空次数: {len(test_short_returns)}")
            print(f"      做空累计收益: {test_short_returns.sum():.6f}")
            print(f"      做空平均收益: {test_short_returns.mean():.6f}")
            print(f"      做空胜率: {(test_short_returns > 0).mean()*100:.2f}%")
        else:
            print(f"      做空次数: 0")
        
        # 5. Sharpe Ratio
        print("\n[8.6] 风险调整收益 - Sharpe Ratio")
        print("-" * 40)
        
        print(f"  【计算说明】")
        if config.DATA_FREQUENCY == 'minute':
            print(f"    数据频率: 分钟数据")
            print(f"    预测周期: {config.PREDICTION_PERIOD_MINUTES}分钟 (未来{config.PREDICTION_PERIOD_MINUTES}期累计收益)")
            print(f"    每天交易分钟数: {config.TRADING_MINUTES_PER_DAY}")
        else:
            print(f"    数据频率: 小时数据")
            print(f"    预测周期: {config.PREDICTION_PERIOD_MINUTES / 60:.2f}小时")
        print(f"    年化因子: {config.PERIODS_PER_YEAR:.2f} (每年预测周期数)")
        
        train_sharpe = self.calculate_sharpe_ratio(train_strategy_returns)
        test_sharpe = self.calculate_sharpe_ratio(test_strategy_returns)
        
        print(f"\n  Sharpe Ratio (年化):")
        print(f"    训练集: {train_sharpe:.2f}")
        print(f"    测试集: {test_sharpe:.2f}")
        
        if test_sharpe > 1.5:
            sharpe_rating = "✓✓ 优秀"
        elif test_sharpe > 1.0:
            sharpe_rating = "✓  良好"
        elif test_sharpe > 0.5:
            sharpe_rating = "⚠  一般"
        else:
            sharpe_rating = "✗  不佳"
        
        print(f"\n  【评级】")
        print(f"    {sharpe_rating}")
        
        print(f"\n  【Sharpe解释】")
        print(f"    Sharpe>2.0: 卓越")
        print(f"    Sharpe>1.5: 优秀")
        print(f"    Sharpe>1.0: 良好")
        print(f"    Sharpe>0.5: 一般")
        print(f"    Sharpe<0.5: 不佳")
        
        # 6. 最大回撤
        print("\n[8.7] 最大回撤分析")
        print("-" * 40)
        
        train_max_dd, train_drawdown, _ = self.calculate_max_drawdown(train_strategy_returns)
        test_max_dd, test_drawdown, test_dd_idx = self.calculate_max_drawdown(test_strategy_returns)
        
        print(f"  最大回撤:")
        print(f"    训练集: {train_max_dd:.6f}")
        print(f"    测试集: {test_max_dd:.6f}")
        print(f"    发生位置: 第{test_dd_idx}个样本")
        if dates_test is not None:
            print(f"    发生时间: {dates_test[test_dd_idx]}")
        
        print(f"\n  【风险评估】")
        if abs(test_max_dd) < abs(test_strategy_returns.sum()) * 0.1:
            print(f"    ✓ 回撤控制良好（<累计收益10%）")
        elif abs(test_max_dd) < abs(test_strategy_returns.sum()) * 0.3:
            print(f"    ⚠ 回撤中等（10-30%累计收益）")
        else:
            print(f"    ✗ 回撤较大（>累计收益30%）")
        
        # 7. 卡玛比率
        print("\n[8.8] 卡玛比率（Calmar Ratio）")
        print("-" * 40)
        
        print(f"  【计算说明】")
        print(f"    卡玛比率 = 累计收益率 / 最大回撤的绝对值")
        print(f"    衡量单位回撤下的收益能力，值越大越好")
        
        train_calmar = self.calculate_calmar_ratio(train_strategy_returns)
        test_calmar = self.calculate_calmar_ratio(test_strategy_returns)
        
        print(f"\n  Calmar Ratio:")
        print(f"    训练集: {train_calmar:.2f}" if not np.isnan(train_calmar) else f"    训练集: N/A")
        print(f"    测试集: {test_calmar:.2f}" if not np.isnan(test_calmar) else f"    测试集: N/A")
        
        if not np.isnan(test_calmar):
            if test_calmar > 3.0:
                calmar_rating = "✓✓ 优秀"
            elif test_calmar > 1.5:
                calmar_rating = "✓  良好"
            elif test_calmar > 0.5:
                calmar_rating = "⚠  一般"
            else:
                calmar_rating = "✗  不佳"
            
            print(f"\n  【评级】")
            print(f"    {calmar_rating}")
            
            print(f"\n  【Calmar解释】")
            print(f"    Calmar>3.0: 优秀（收益远超回撤）")
            print(f"    Calmar>1.5: 良好（收益明显大于回撤）")
            print(f"    Calmar>0.5: 一般（收益略大于回撤）")
            print(f"    Calmar<0.5: 不佳（收益小于回撤）")
        else:
            calmar_rating = None
            print(f"\n  【说明】最大回撤为0，无法计算卡玛比率")
        
        # 8. 分位数分析
        print("\n[8.9] 预测值分位数分析")
        print("-" * 40)
        
        print(f"  【目的】验证预测值大小与实际收益的单调关系")
        
        quantile_stats = self.calculate_quantile_analysis(y_test, y_test_pred)
        
        print(f"\n  分位数统计:")
        print(quantile_stats)
        
        q5_q1_diff = quantile_stats['mean'].iloc[-1] - quantile_stats['mean'].iloc[0]
        print(f"\n  Q5 vs Q1 平均收益差: {q5_q1_diff:.6f}")
        if q5_q1_diff > 0:
            print(f"    ✓ 正向单调关系（预测值越大，实际收益越高）")
        else:
            print(f"    ✗ 单调关系不成立")
        
        # 汇总所有结果
        results = {
            'train': {
                'rmse': train_metrics['rmse'],
                'mae': train_metrics['mae'],
                'ic': train_corr['ic'],
                'rank_ic': train_corr['rank_ic'],
                'direction_acc': train_dir_acc,
                'sharpe': train_sharpe,
                'calmar': train_calmar,
                'cum_return': train_strategy_returns.sum(),
                'max_drawdown': train_max_dd,
                'win_rate': (train_strategy_returns > 0).mean(),
                'win_loss_ratio': win_loss_ratio_train
            },
            'test': {
                'rmse': test_metrics['rmse'],
                'mae': test_metrics['mae'],
                'ic': test_corr['ic'],
                'rank_ic': test_corr['rank_ic'],
                'direction_acc': test_dir_acc,
                'sharpe': test_sharpe,
                'calmar': test_calmar,
                'cum_return': test_strategy_returns.sum(),
                'max_drawdown': test_max_dd,
                'win_rate': (test_strategy_returns > 0).mean(),
                'win_loss_ratio': win_loss_ratio_test
            },
            'confusion_matrix': test_cm,
            'quantile_stats': quantile_stats,
            'test_strategy_returns': test_strategy_returns,
            'test_drawdown': test_drawdown,
            'rating': rating,
            'sharpe_rating': sharpe_rating,
            'calmar_rating': calmar_rating,
            'q5_q1_diff': q5_q1_diff
        }
        
        return results

