import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from lib import logger

from lib.cux001 import FactorEvaluate1

class Evaluator(object):
    """
    单品种时序预测的高斯NLL模型评估类

    核心评估指标:
    - prediction: 原始预测值 (模型对预期收益率的直接估计)
    - adjusted_prediction = prediction / sqrt(variance): 风险调整后的信号

    关键验证:
    - 如果 adjusted_prediction 的 IC > prediction 的 IC
      → 证明方差预测有效，模型成功识别了不可靠的预测并降低了权重
    """

    def __init__(
        self,
        fee: float = 0.0,
        resampling_win: int = 1,
        roll_win: int = 252,
        scale_method: str = "raw",
    ):

        self.fee = fee
        self.resampling_win = max(1, int(resampling_win))
        self.roll_win = max(5, int(roll_win))
        self.scale_method = scale_method

    def calculate_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算IC (Pearson相关系数)"""
        return np.corrcoef(y_pred, y_true)[0, 1]
    
    def calculate_rank_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算RankIC (Spearman相关系数)"""
        rank_ic, _ = spearmanr(y_pred, y_true)
        return rank_ic

    def calculate_adjusted_prediction(
        self,
        prediction: np.ndarray,
        variance: np.ndarray
    ) -> np.ndarray:
        """
        计算风险调整后的预测值

        adjusted_prediction = prediction / sqrt(variance)

        含义: 类似夏普比率，高方差(不确定)的预测权重降低
        """
        std_dev = np.sqrt(np.maximum(variance, 1e-6))
        return prediction / std_dev
    
    def calculate_direction_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """计算方向准确率和混淆矩阵"""
        pred_direction = np.sign(y_pred)
        true_direction = np.sign(y_true)
        direction_acc = np.mean(pred_direction == true_direction)
        cm = confusion_matrix(true_direction, pred_direction, labels=[-1, 1])
        return direction_acc, cm
    
    def calculate_gaussian_nll(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        variance: np.ndarray,
    ) -> float:
        """
        计算高斯负对数似然损失

        L = 0.5 * (log(σ²) + (y - μ)² / σ²)
        """
        variance = np.maximum(variance, 1e-6)
        nll = 0.5 * np.mean(np.log(variance) + (y_true - y_pred) ** 2 / variance)
        return nll
    
    # ==================== 方差有效性验证 ====================

    def verify_variance_effectiveness(
        self,
        y_true: np.ndarray,
        prediction: np.ndarray,
        variance: np.ndarray,
    ) -> Dict:
        """
        验证方差预测的有效性

        核心问题: adjusted_prediction 的 IC 是否 > prediction 的 IC?

        Returns:
            Dict containing:
            - prediction_ic: 原始预测的IC
            - prediction_rank_ic: 原始预测的RankIC
            - adjusted_ic: 风险调整后的IC
            - adjusted_rank_ic: 风险调整后的RankIC
            - variance_is_effective: bool, 方差预测是否有效
            - ic_improvement: IC提升百分比
        """
        # 原始预测指标
        pred_ic = self.calculate_ic(y_true, prediction)
        pred_rank_ic = self.calculate_rank_ic(y_true, prediction)

        # 风险调整后的预测指标
        adjusted_pred = self.calculate_adjusted_prediction(prediction, variance)
        adj_ic = self.calculate_ic(y_true, adjusted_pred)
        adj_rank_ic = self.calculate_rank_ic(y_true, adjusted_pred)

        # 判断方差是否有效
        variance_is_effective = adj_ic > pred_ic

        # 计算IC提升
        if pred_ic != 0:
            ic_improvement = (adj_ic - pred_ic) / abs(pred_ic) * 100
        else:
            ic_improvement = np.nan

        return {
            'prediction_ic': pred_ic,
            'prediction_rank_ic': pred_rank_ic,
            'adjusted_ic': adj_ic,
            'adjusted_rank_ic': adj_rank_ic,
            'variance_is_effective': variance_is_effective,
            'ic_improvement': ic_improvement,
        }
    
    # ==================== 混淆矩阵展示 ====================

    def show_matrix(self, cm: np.ndarray, title: str = "混淆矩阵"):
        """展示混淆矩阵"""
        logger.panel(
            f"(左上角) 预测下跌且实际下跌 - 正确识别熊市\n"
            f"(右下角) 预测上涨且实际上涨 - 正确识别牛市\n"
            f"(右上角) 预测上涨但实际下跌 - 危险! 假正例, 导致亏损\n"
            f"(左下角) 预测下跌但实际上涨 - 机会成本, 错过盈利",
            title="混淆矩阵说明"
        )

        cm_df = pd.DataFrame(
            cm,
            index=["实际下跌", "实际上涨"],
            columns=["预测下跌", "预测上涨"]
        )
        cm_df.index.name = ""
        logger.table(cm_df.reset_index(), title=title)

        total = cm.sum()
        prob_table = pd.DataFrame([
            ("正确预测上涨", cm[1, 1], f"{cm[1, 1]/total:.2%}"),
            ("正确预测下跌", cm[0, 0], f"{cm[0, 0]/total:.2%}"),
            ("错误预测下跌(假负)", cm[1, 0], f"{cm[1, 0]/total:.2%}"),
            ("错误预测上涨(假正)", cm[0, 1], f"{cm[0, 1]/total:.2%}"),
        ], columns=["类型", "样本数", "占比"])
        logger.table(prob_table, title="方向预测统计")

    # ==================== 训练集+校验集评估 ====================
        
    def fitting_evaluate(
        self,
        y_train_true: np.ndarray,
        y_train_pred: np.ndarray,
        y_val_true: np.ndarray,
        y_val_pred: np.ndarray,
        var_train: np.ndarray,
        var_val: np.ndarray,
    ) -> Dict:
        """
        训练集+校验集评估 - 用于调整超参

        核心评估:
        1. 基础预测能力: IC, RankIC, 方向准确率
        2. 方差有效性验证: adjusted_prediction IC vs prediction IC
        3. 损失函数: Gaussian NLL

        Returns:
            包含训练集和校验集评估结果的Dict
        """
        logger.rule("模型评估 (训练集 + 校验集)")

        results = {'train': {}, 'val': {}}

        # ========== 训练集评估 ==========
        train_verify = self.verify_variance_effectiveness(
            y_train_true, y_train_pred, var_train
        )
        train_dir_acc, train_cm = self.calculate_direction_accuracy(
            y_train_true, y_train_pred
        )
        train_nll = self.calculate_gaussian_nll(
            y_train_true, y_train_pred, var_train
        )

        results['train'] = {
            **train_verify,
            'direction_acc': train_dir_acc,
            'gaussian_nll': train_nll,
            'var_mean': np.mean(var_train),
            'var_std': np.std(var_train),
        }

        # ========== 校验集评估 ==========
        val_verify = self.verify_variance_effectiveness(
            y_val_true, y_val_pred, var_val
        )
        val_dir_acc, val_cm = self.calculate_direction_accuracy(
            y_val_true, y_val_pred
        )
        val_nll = self.calculate_gaussian_nll(
            y_val_true, y_val_pred, var_val
        )

        results['val'] = {
            **val_verify,
            'direction_acc': val_dir_acc,
            'gaussian_nll': val_nll,
            'var_mean': np.mean(var_val),
            'var_std': np.std(var_val),
        }

        # ========== 输出结果 ==========
        self._display_fitting_results(results, val_cm)

        #return results

    def final_evaluate(
        self,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
        var_test: np.ndarray,
        dates_test: np.ndarray,
        returns: pd.Series,
        period: int,
    ) -> Dict:
        """
        测试集评估 - 侧重策略评估

        使用 FactorEvaluate1 评估:
        1. 原始预测 (prediction) 作为因子
        2. 风险调整预测 (adjusted_prediction) 作为因子

        比较两者的策略表现，验证方差预测的实际价值

        Args:
            y_test_true: 测试集真实值
            y_test_pred: 测试集预测值 (prediction)
            var_test: 测试集方差
            dates_test: 测试集日期
            returns: 收益率序列 (需包含 nxt1_ret_{period}h 列)
            period: 预测周期

        Returns:
            包含策略评估结果的Dict
        """
        logger.rule("测试集评估 (策略评估)")

        results = {}

        # ========== 1. 基础预测能力评估 ==========
        verify_results = self.verify_variance_effectiveness(
            y_test_true, y_test_pred, var_test
        )
        dir_acc, test_cm = self.calculate_direction_accuracy(
            y_test_true, y_test_pred
        )
        nll = self.calculate_gaussian_nll(y_test_true, y_test_pred, var_test)

        results['basic'] = {
            **verify_results,
            'direction_acc': dir_acc,
            'gaussian_nll': nll,
        }

        # 显示基础指标
        self._display_test_basic_results(results['basic'], test_cm)

        # ========== 2. 策略评估 (FactorEvaluate1) ==========
        adjusted_pred = self.calculate_adjusted_prediction(y_test_pred, var_test)

        # 构建因子DataFrame
        factor_df = pd.DataFrame({
            'trade_time': dates_test,
            'prediction': y_test_pred,
            'adjusted_prediction': adjusted_pred,
        })
        factor_df = factor_df.set_index('trade_time')

        # 评估原始预测因子
        #logger.rule("策略评估: 原始预测 (prediction)")
        pred_stats, pred_returns = self._evaluate_factor(
            factor_df[['prediction']].rename(columns={'prediction': 'transformed'}),
            returns, period, "prediction"
        )
        results['prediction_strategy'] = {'stats': pred_stats, 'returns': pred_returns}

        # 评估风险调整预测因子
        #logger.rule("策略评估: 风险调整预测 (adjusted_prediction)")
        adj_stats, adj_returns = self._evaluate_factor(
            factor_df[['adjusted_prediction']].rename(columns={'adjusted_prediction': 'transformed'}),
            returns, period, "adjusted_prediction"
        )
        results['adjusted_strategy'] = {'stats': adj_stats, 'returns': adj_returns}

        # ========== 3. 策略对比 ==========
        self._display_strategy_comparison(pred_stats, adj_stats, pred_returns, adj_returns)

    def _evaluate_factor(
        self,
        factors: pd.DataFrame,
        returns: pd.Series,
        period: int,
        name: str
    ) -> Tuple[Dict, Dict]:
        """使用FactorEvaluate1评估因子"""
        data = pd.merge(
            factors, returns,
            how='left',
            left_index=True,
            right_index=True
        )

        evaluate = FactorEvaluate1(
            factor_data=data.reset_index(),
            factor_name='transformed',
            ret_name=f'nxt1_ret_{period}h',
            roll_win=self.roll_win,
            fee=self.fee,
            scale_method=self.scale_method,
            expression="test",
            resampling_win=self.resampling_win
        )

        stats = evaluate.run()
        factor_returns = evaluate.cal_returns()

        stats['name'] = name
        factor_returns['name'] = name

        return stats, factor_returns
    

    def _display_fitting_results(self, results: Dict, val_cm: np.ndarray):
        """展示训练+校验集评估结果"""
        train = results['train']
        val = results['val']

        # 1. 核心指标对比: prediction vs adjusted_prediction
        logger.panel(
            f"【原始预测 (prediction)】\n"
            f"  IC       训练/校验: {train['prediction_ic']:.4f} / {val['prediction_ic']:.4f}\n"
            f"  RankIC   训练/校验: {train['prediction_rank_ic']:.4f} / {val['prediction_rank_ic']:.4f}\n"
            f"\n"
            f"【风险调整后 (prediction / sqrt(variance))】\n"
            f"  IC       训练/校验: {train['adjusted_ic']:.4f} / {val['adjusted_ic']:.4f}\n"
            f"  RankIC   训练/校验: {train['adjusted_rank_ic']:.4f} / {val['adjusted_rank_ic']:.4f}\n"
            f"\n"
            f"【IC提升】\n"
            f"  训练集: {train['ic_improvement']:+.2f}%\n"
            f"  校验集: {val['ic_improvement']:+.2f}%",
            title="核心指标: 原始预测 vs 风险调整预测"
        )

        # 2. 方差有效性判断
        train_effective = "✓ 有效" if train['variance_is_effective'] else "✗ 无效"
        val_effective = "✓ 有效" if val['variance_is_effective'] else "✗ 无效"

        logger.panel(
            f"  训练集方差预测: {train_effective}\n"
            f"  校验集方差预测: {val_effective}\n"
            f"\n"
            f"  判断依据: adjusted_prediction IC > prediction IC\n"
            f"  含义: 方差预测成功识别了不可靠的预测并降低了权重",
            title="方差有效性验证"
        )

        # 3. 其他指标
        logger.panel(
            f"  方向准确率 训练/校验: {train['direction_acc']*100:.2f}% / {val['direction_acc']*100:.2f}%\n"
            f"  Gaussian NLL 训练/校验: {train['gaussian_nll']:.6f} / {val['gaussian_nll']:.6f}\n"
            f"  方差均值 训练/校验: {train['var_mean']:.6f} / {val['var_mean']:.6f}\n"
            f"  方差标准差 训练/校验: {train['var_std']:.6f} / {val['var_std']:.6f}",
            title="辅助指标"
        )

        # 4. 混淆矩阵 (校验集)
        self.show_matrix(val_cm, title="混淆矩阵 (校验集)")

    def _display_test_basic_results(self, basic: Dict, test_cm: np.ndarray):
        """展示测试集基础评估结果"""
        logger.panel(
            f"【原始预测 (prediction)】\n"
            f"  IC: {basic['prediction_ic']:.4f}\n"
            f"  RankIC: {basic['prediction_rank_ic']:.4f}\n"
            f"\n"
            f"【风险调整后 (prediction / sqrt(variance))】\n"
            f"  IC: {basic['adjusted_ic']:.4f}\n"
            f"  RankIC: {basic['adjusted_rank_ic']:.4f}\n"
            f"\n"
            f"【IC提升】: {basic['ic_improvement']:+.2f}%\n"
            f"【方差有效性】: {'✓ 有效' if basic['variance_is_effective'] else '✗ 无效'}",
            title="测试集: 预测能力评估"
        )

        logger.panel(
            f"  方向准确率: {basic['direction_acc']*100:.2f}%\n"
            f"  Gaussian NLL: {basic['gaussian_nll']:.6f}",
            title="测试集: 辅助指标"
        )

        self.show_matrix(test_cm, title="混淆矩阵 (测试集)")

    def _display_strategy_comparison(
        self,
        pred_stats: Dict,
        adj_stats: Dict,
        pred_returns: Dict,
        adj_returns: Dict
    ):
        """展示策略对比结果"""
        logger.rule("策略对比: prediction vs adjusted_prediction")

        # 构建对比表格
        comparison_rows = []

        # 从stats中提取关键指标
        for key in pred_stats.keys():
            if key != 'name' and isinstance(pred_stats[key], (int, float)):
                pred_val = pred_stats[key]
                adj_val = adj_stats.get(key, np.nan)
                if not np.isnan(pred_val) and not np.isnan(adj_val):
                    diff = adj_val - pred_val
                    comparison_rows.append((
                        key,
                        f"{pred_val:.4f}" if abs(pred_val) < 100 else f"{pred_val:.2f}",
                        f"{adj_val:.4f}" if abs(adj_val) < 100 else f"{adj_val:.2f}",
                        f"{diff:+.4f}" if abs(diff) < 100 else f"{diff:+.2f}",
                    ))

        if comparison_rows:
            comparison_df = pd.DataFrame(
                comparison_rows,
                columns=["指标", "prediction", "adjusted_prediction", "差值"]
            )
            logger.table(comparison_df, title="策略绩效对比")

        # 收益对比
        returns_rows = []
        for key in pred_returns.keys():
            if key != 'name' and isinstance(pred_returns[key], (int, float)):
                pred_val = pred_returns[key]
                adj_val = adj_returns.get(key, np.nan)
                if not np.isnan(pred_val) and not np.isnan(adj_val):
                    returns_rows.append((
                        key,
                        f"{pred_val:.4f}" if abs(pred_val) < 100 else f"{pred_val:.2f}",
                        f"{adj_val:.4f}" if abs(adj_val) < 100 else f"{adj_val:.2f}",
                    ))

        if returns_rows:
            returns_df = pd.DataFrame(
                returns_rows,
                columns=["指标", "prediction", "adjusted_prediction"]
            )
            logger.table(returns_df, title="多空收益对比")

        # 结论
        logger.panel(
            f"如果 adjusted_prediction 策略表现优于 prediction 策略:\n"
            f"  → 证明方差预测在实际策略中有价值\n"
            f"  → 模型成功识别了不可靠的预测并降低了权重\n"
            f"  → 建议在实盘中使用风险调整后的信号",
            title="结论"
        )
