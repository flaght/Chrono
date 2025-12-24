import pdb
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
        roll_wins: list = None,
        scale_method: str = "raw",
        output_dir: str = None,
        model_id: str = None,
        save_plots: bool = False,
    ):

        self.fee = fee
        self.resampling_win = max(1, int(resampling_win))
        # 支持多个 roll_win 评估，默认 [120]
        self.roll_wins = roll_wins if roll_wins is not None else [120]
        self.scale_method = scale_method

        self.output_dir = output_dir
        self.model_id = model_id
        self.save_plots = save_plots

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
        dates_train: np.ndarray = None,
        dates_val: np.ndarray = None,
        returns: pd.Series = None,
        period: int = None,
    ) -> Dict:
        """
        训练集+校验集评估 - 用于调整超参

        核心评估:
        1. 基础预测能力: IC, RankIC, 方向准确率
        2. 方差有效性验证: adjusted_prediction IC vs prediction IC
        3. 损失函数: Gaussian NLL
        4. 滚动评估 (FactorEvaluate1): 与测试集保持一致的评估方法

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
        
        logger.rule("滚动评估 (训练集 + 校验集) - 多 Roll_Win")
        train_adjusted_pred = self.calculate_adjusted_prediction(y_train_pred, var_train)
        train_factor_df = pd.DataFrame({
                'trade_time': dates_train,
                'prediction': y_train_pred.flatten(),
                'adjusted_prediction': train_adjusted_pred.flatten(),
            }).set_index('trade_time')

        # 校验集数据准备
        val_adjusted_pred = self.calculate_adjusted_prediction(y_val_pred, var_val)
        val_factor_df = pd.DataFrame({
                'trade_time': dates_val,
                'prediction': y_val_pred.flatten(),
                'adjusted_prediction': val_adjusted_pred.flatten(),
            }).set_index('trade_time')

        # 对每个 roll_win 进行评估
        all_train_stats = []
        all_val_stats = []

        logger.rule("原始预测评估 (不做roll标准化)")

        train_pred_raw_stats, _ = self._evaluate_factor(
            train_factor_df[['prediction']].rename(columns={'prediction': 'transformed'}),
            returns, period, "train_prediction_raw", roll_win=15, scale_method='raw'
        )
        train_adj_raw_stats, _ = self._evaluate_factor(
            train_factor_df[['adjusted_prediction']].rename(columns={'adjusted_prediction': 'transformed'}),
            returns, period, "train_adjusted_raw", roll_win=15, scale_method='raw'
        )

        val_pred_raw_stats, _ = self._evaluate_factor(
            val_factor_df[['prediction']].rename(columns={'prediction': 'transformed'}),
            returns, period, "val_prediction_raw", roll_win=15, scale_method='raw'
        )
        val_adj_raw_stats, _ = self._evaluate_factor(
            val_factor_df[['adjusted_prediction']].rename(columns={'adjusted_prediction': 'transformed'}),
            returns, period, "val_adjusted_raw", roll_win=15, scale_method='raw'
        )

        self._display_fitting_raw_evaluation(
            train_pred_raw_stats, train_adj_raw_stats,
            val_pred_raw_stats, val_adj_raw_stats
        )

        for roll_win in self.roll_wins:
            # 训练集评估
            train_pred_stats, train_pred_returns = self._evaluate_factor(
                    train_factor_df[['prediction']].rename(columns={'prediction': 'transformed'}),
                    returns, period, "train_prediction", roll_win=roll_win
                )
            all_train_stats.append(train_pred_stats)

            # 校验集评估
            val_pred_stats, val_pred_returns = self._evaluate_factor(
                    val_factor_df[['prediction']].rename(columns={'prediction': 'transformed'}),
                    returns, period, "val_prediction", roll_win=roll_win
                )
            all_val_stats.append(val_pred_stats)

        # 保存第一个 roll_win 的结果
        results['train']['strategy_stats'] = all_train_stats[0]
        results['train']['strategy_returns'] = train_pred_returns
        results['val']['strategy_stats'] = all_val_stats[0]
        results['val']['strategy_returns'] = val_pred_returns

        # 展示多 roll_win 对比
        self._display_fitting_multi_rollwin(all_train_stats, all_val_stats)

        #return results
    def _display_rolling_comparison(
        self,
        train_stats: Dict,
        val_stats: Dict,
        train_returns: Dict,
        val_returns: Dict
    ):
        """展示训练集/校验集滚动评估对比"""
        # 构建对比表格
        comparison_rows = []

        # 关键指标
        key_metrics = ['ic_mean', 'ic_std', 'ic_ir', 'annual_return', 'sharpe', 'calmar', 'max_drawdown']

        for key in key_metrics:
            train_val = train_stats.get(key, np.nan)
            val_val = val_stats.get(key, np.nan)
            if not np.isnan(train_val) and not np.isnan(val_val):
                comparison_rows.append((
                    key,
                    f"{train_val:.4f}" if abs(train_val) < 100 else f"{train_val:.2f}",
                    f"{val_val:.4f}" if abs(val_val) < 100 else f"{val_val:.2f}",
                ))

        if comparison_rows:
            comparison_df = pd.DataFrame(
                comparison_rows,
                columns=["指标", "训练集", "校验集"]
            )
            logger.table(comparison_df, title="滚动评估对比 (FactorEvaluate1)")

    def final_evaluate(
        self,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
        var_test: np.ndarray,
        dates_test: np.ndarray,
        returns: pd.Series,
        period: int,
        model=None,  # 新增: 用于参数诊断
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
            model: 模型对象，用于参数诊断 (可选)

        Returns:
            包含策略评估结果的Dict
        """
        logger.rule("测试集评估 (策略评估)")

        # ========== 新增: 模型参数诊断 ==========
        if model is not None:
            self._display_model_diagnosis(model)

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

        # ========== 2. 策略评估 (FactorEvaluate1) - 多 roll_win ==========
        adjusted_pred = self.calculate_adjusted_prediction(y_test_pred, var_test)

        # 构建因子DataFrame
        factor_df = pd.DataFrame({
            'trade_time': dates_test,
            'prediction': y_test_pred,
            'adjusted_prediction': adjusted_pred,
        })
        factor_df = factor_df.set_index('trade_time')

        # 对每个 roll_win 进行评估
        all_pred_stats = []
        all_adj_stats = []

        # ========== 新增: 预测值分布统计 ==========
        self._display_prediction_distribution(y_test_pred, adjusted_pred, y_test_true)

        # ========== 新增: 原始预测评估 (不roll) ==========
        logger.rule("原始预测评估 (不做roll标准化)")

        pred_raw_stats, pred_raw_returns = self._evaluate_factor(
            factor_df[['prediction']].rename(columns={'prediction': 'transformed'}),
            returns, period, "prediction_raw", roll_win=15, scale_method='raw'
        )

        adj_raw_stats, adj_raw_returns = self._evaluate_factor(
            factor_df[['adjusted_prediction']].rename(columns={'adjusted_prediction': 'transformed'}),
            returns, period, "adjusted_prediction_raw", roll_win=15, scale_method='raw'
        )

        self._display_raw_evaluation(pred_raw_stats, adj_raw_stats)

        # 设置 raw 的 roll_win 标识为 'raw'，便于在对比表中区分
        pred_raw_stats['roll_win'] = 'raw'
        adj_raw_stats['roll_win'] = 'raw'

        results['prediction_raw'] = {'stats': pred_raw_stats, 'returns': pred_raw_returns}
        results['adjusted_raw'] = {'stats': adj_raw_stats, 'returns': adj_raw_returns}

        # ========== Roll标准化评估 ==========
        logger.rule("Roll标准化评估")

        for roll_win in self.roll_wins:
            # 评估原始预测因子
            pred_stats, pred_returns = self._evaluate_factor(
                factor_df[['prediction']].rename(columns={'prediction': 'transformed'}),
                returns, period, "prediction", roll_win=roll_win
            )
            all_pred_stats.append(pred_stats)

            # 评估风险调整预测因子
            adj_stats, adj_returns = self._evaluate_factor(
                factor_df[['adjusted_prediction']].rename(columns={'adjusted_prediction': 'transformed'}),
                returns, period, "adjusted_prediction", roll_win=roll_win
            )
            all_adj_stats.append(adj_stats)

        # 保存结果（使用第一个 roll_win 作为默认）
        results['prediction_strategy'] = {'stats': all_pred_stats[0], 'returns': pred_returns}
        results['adjusted_strategy'] = {'stats': all_adj_stats[0], 'returns': adj_returns}

        # ========== 3. 多 roll_win 策略对比 (包含 raw) ==========
        # 将 raw 结果添加到对比列表的开头
        all_pred_stats_with_raw = [pred_raw_stats] + all_pred_stats
        all_adj_stats_with_raw = [adj_raw_stats] + all_adj_stats
        self._display_multi_rollwin_comparison(all_pred_stats_with_raw, all_adj_stats_with_raw)

    def _evaluate_factor(
        self,
        factors: pd.DataFrame,
        returns: pd.Series,
        period: int,
        name: str,
        roll_win: int = 120,
        scale_method: str = None,
    ) -> Tuple[Dict, Dict]:
        """使用FactorEvaluate1评估因子"""
        data = pd.merge(
            factors, returns,
            how='left',
            left_index=True,
            right_index=True
        )
        actual_scale_method = scale_method if scale_method else self.scale_method

        evaluate = FactorEvaluate1(
            factor_data=data.reset_index(),
            factor_name='transformed',
            ret_name=f'nxt1_ret_{period}h',
            roll_win=roll_win,
            fee=self.fee,
            scale_method=actual_scale_method,
            expression="test",
            resampling_win=self.resampling_win
        )

        stats = evaluate.run()
        factor_returns = evaluate.cal_returns()
        # 保存图表
        if self.save_plots and self.output_dir and self.model_id:
            self._save_evaluation_plot(evaluate, name, actual_scale_method, roll_win)

        stats['name'] = name
        stats['roll_win'] = roll_win
        factor_returns['name'] = name

        return stats, factor_returns
    

    def _save_evaluation_plot(
        self,
        evaluate,
        factor_name: str,
        scale_method: str,
        roll_win: int
    ):
        """保存评估图表"""
        import os

        factor_type = 'adj_pred' if 'adjusted' in factor_name else 'pred'

        if scale_method == 'raw':
            subdir = f"{factor_type}_raw"
        else:
            subdir = f"{factor_type}_{scale_method}_{roll_win}"

        plot_dir = os.path.join(self.output_dir, 'evaluation', self.model_id, subdir)
        os.makedirs(plot_dir, exist_ok=True)

        try:
            evaluate.plot_results()
            evaluate.save_results(plot_dir)
            logger.print(f"图表已保存至: {plot_dir}")
        except Exception as e:
            logger.print(f"保存图表失败: {e}")

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

    def _display_multi_rollwin_comparison(
        self,
        all_pred_stats: list,
        all_adj_stats: list
    ):
        """展示多 roll_win 的策略对比"""
        logger.rule("多 Roll_Win 策略对比")

        # 关键指标列表
        key_metrics = ['ic_mean', 'ic_ir', 'total_ret', 'calmar', 'sharpe2', 'turnover', 'profit_ratio']

        # 构建 prediction 对比表
        pred_rows = []
        for stats in all_pred_stats:
            row = {'roll_win': stats.get('roll_win', '-')}
            for metric in key_metrics:
                val = stats.get(metric, np.nan)
                if not np.isnan(val):
                    row[metric] = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
                else:
                    row[metric] = "-"
            pred_rows.append(row)

        if pred_rows:
            pred_df = pd.DataFrame(pred_rows)
            logger.table(pred_df, title="Prediction 策略 (不同 roll_win)")

        # 构建 adjusted_prediction 对比表
        adj_rows = []
        for stats in all_adj_stats:
            row = {'roll_win': stats.get('roll_win', '-')}
            for metric in key_metrics:
                val = stats.get(metric, np.nan)
                if not np.isnan(val):
                    row[metric] = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
                else:
                    row[metric] = "-"
            adj_rows.append(row)

        if adj_rows:
            adj_df = pd.DataFrame(adj_rows)
            logger.table(adj_df, title="Adjusted Prediction 策略 (不同 roll_win)")

    def _display_fitting_multi_rollwin(
        self,
        all_train_stats: list,
        all_val_stats: list
    ):
        """展示训练集/校验集在多 roll_win 下的对比"""
        # 关键指标
        key_metrics = ['ic_mean', 'ic_ir', 'total_ret', 'calmar', 'sharpe2']

        # 构建对比表：每行一个 roll_win，列为 train/val 对比
        rows = []
        for train_stats, val_stats in zip(all_train_stats, all_val_stats):
            roll_win = train_stats.get('roll_win', '-')
            row = {'roll_win': roll_win}

            for metric in key_metrics:
                train_val = train_stats.get(metric, np.nan)
                val_val = val_stats.get(metric, np.nan)

                if not np.isnan(train_val) and not np.isnan(val_val):
                    # 判断方向是否一致
                    same_sign = (train_val * val_val) > 0
                    sign_indicator = "✓" if same_sign else "✗"

                    train_str = f"{train_val:.4f}" if abs(train_val) < 100 else f"{train_val:.2f}"
                    val_str = f"{val_val:.4f}" if abs(val_val) < 100 else f"{val_val:.2f}"
                    row[f"{metric}_train"] = train_str
                    row[f"{metric}_val"] = val_str
                    row[f"{metric}_sign"] = sign_indicator
                else:
                    row[f"{metric}_train"] = "-"
                    row[f"{metric}_val"] = "-"
                    row[f"{metric}_sign"] = "-"

            rows.append(row)

        if rows:
            # 简化显示：只显示关键指标的 train vs val
            simple_rows = []
            for train_stats, val_stats in zip(all_train_stats, all_val_stats):
                roll_win = train_stats.get('roll_win', '-')

                train_ic = train_stats.get('ic_mean', np.nan)
                val_ic = val_stats.get('ic_mean', np.nan)
                train_calmar = train_stats.get('calmar', np.nan)
                val_calmar = val_stats.get('calmar', np.nan)

                # 方向一致性检查
                ic_same = "✓" if (train_ic * val_ic) > 0 else "✗"
                calmar_same = "✓" if (train_calmar * val_calmar) > 0 else "✗"

                simple_rows.append({
                    'roll_win': roll_win,
                    'IC_train': f"{train_ic:.4f}" if not np.isnan(train_ic) else "-",
                    'IC_val': f"{val_ic:.4f}" if not np.isnan(val_ic) else "-",
                    'IC_一致': ic_same,
                    'Calmar_train': f"{train_calmar:.2f}" if not np.isnan(train_calmar) else "-",
                    'Calmar_val': f"{val_calmar:.2f}" if not np.isnan(val_calmar) else "-",
                    'Calmar_一致': calmar_same,
                })

            simple_df = pd.DataFrame(simple_rows)
            logger.table(simple_df, title="训练集 vs 校验集 (多 Roll_Win 对比)")


    def _display_raw_evaluation(self, pred_stats: Dict, adj_stats: Dict):
        """展示原始预测评估结果 (不roll)"""
        key_metrics = ['ic_mean', 'ic_ir', 'total_ret', 'calmar', 'sharpe2', 'turnover']

        rows = []
        for metric in key_metrics:
            pred_val = pred_stats.get(metric, np.nan)
            adj_val = adj_stats.get(metric, np.nan)

            if not np.isnan(pred_val) and not np.isnan(adj_val):
                diff = adj_val - pred_val
                rows.append({
                    '指标': metric,
                    'prediction': f"{pred_val:.4f}" if abs(pred_val) < 100 else f"{pred_val:.2f}",
                    'adjusted_prediction': f"{adj_val:.4f}" if abs(adj_val) < 100 else f"{adj_val:.2f}",
                    '差值': f"{diff:+.4f}" if abs(diff) < 100 else f"{diff:+.2f}",
                })

        if rows:
            df = pd.DataFrame(rows)
            logger.table(df, title="原始预测评估 (scale_method='raw', 不做roll标准化)")

    def _display_fitting_raw_evaluation(
        self,
        train_pred_stats: Dict,
        train_adj_stats: Dict,
        val_pred_stats: Dict,
        val_adj_stats: Dict
    ):
        """展示训练集/校验集原始预测评估结果"""
        key_metrics = ['ic_mean', 'total_ret', 'calmar', 'sharpe2', 'turnover']

        rows = []
        for metric in key_metrics:
            train_pred = train_pred_stats.get(metric, np.nan)
            train_adj = train_adj_stats.get(metric, np.nan)
            val_pred = val_pred_stats.get(metric, np.nan)
            val_adj = val_adj_stats.get(metric, np.nan)

            def fmt(v):
                if np.isnan(v):
                    return "-"
                return f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"

            rows.append({
                '指标': metric,
                'train_pred': fmt(train_pred),
                'train_adj': fmt(train_adj),
                'val_pred': fmt(val_pred),
                'val_adj': fmt(val_adj),
            })

        if rows:
            df = pd.DataFrame(rows)
            logger.table(df, title="原始预测 vs 方差调整 (scale_method='raw', 不做roll标准化)")

    def _display_prediction_distribution(
        self,
        prediction: np.ndarray,
        adjusted_prediction: np.ndarray,
        y_true: np.ndarray = None
    ):
        """展示预测值的分布统计，用于诊断模型偏置"""
        logger.rule("预测值分布统计 (诊断模型偏置)")

        pred = prediction.flatten()
        adj_pred = adjusted_prediction.flatten()

        # 如果有真实值，先展示目标分布
        if y_true is not None:
            y = y_true.flatten()
            y_stats = {
                '均值': np.mean(y),
                '标准差': np.std(y),
                '最小值': np.min(y),
                '最大值': np.max(y),
                '中位数': np.median(y),
                '正值比例': (y > 0).mean(),
                '负值比例': (y < 0).mean(),
            }

            y_rows = []
            for key, val in y_stats.items():
                if '比例' in key:
                    y_rows.append({'统计量': key, '目标y (真实值)': f"{val:.2%}"})
                else:
                    y_rows.append({'统计量': key, '目标y (真实值)': f"{val:.6f}"})

            y_df = pd.DataFrame(y_rows)
            logger.table(y_df, title="目标变量 y 分布统计")

        # 计算统计量
        pred_stats = {
            '均值': np.mean(pred),
            '标准差': np.std(pred),
            '最小值': np.min(pred),
            '最大值': np.max(pred),
            '中位数': np.median(pred),
            '正值比例': (pred > 0).mean(),
            '负值比例': (pred < 0).mean(),
            '零值比例': (pred == 0).mean(),
        }

        adj_stats = {
            '均值': np.mean(adj_pred),
            '标准差': np.std(adj_pred),
            '最小值': np.min(adj_pred),
            '最大值': np.max(adj_pred),
            '中位数': np.median(adj_pred),
            '正值比例': (adj_pred > 0).mean(),
            '负值比例': (adj_pred < 0).mean(),
            '零值比例': (adj_pred == 0).mean(),
        }

        # 构建对比表
        rows = []
        for key in pred_stats.keys():
            pred_val = pred_stats[key]
            adj_val = adj_stats[key]

            if '比例' in key:
                rows.append({
                    '统计量': key,
                    'prediction': f"{pred_val:.2%}",
                    'adjusted_prediction': f"{adj_val:.2%}",
                })
            else:
                rows.append({
                    '统计量': key,
                    'prediction': f"{pred_val:.6f}",
                    'adjusted_prediction': f"{adj_val:.6f}",
                })

        df = pd.DataFrame(rows)
        logger.table(df, title="预测值分布统计")

        # 诊断信息
        bias_warning = ""
        if pred_stats['正值比例'] > 0.9:
            bias_warning = "⚠️ 预测值 90%+ 为正，模型可能存在正向偏置"
        elif pred_stats['负值比例'] > 0.9:
            bias_warning = "⚠️ 预测值 90%+ 为负，模型可能存在负向偏置"
        elif abs(pred_stats['均值']) > 3 * pred_stats['标准差']:
            bias_warning = "⚠️ 均值显著偏离0，模型可能存在偏置"
        else:
            bias_warning = "✓ 预测值分布正常，无明显偏置"

        logger.panel(
            f"{bias_warning}\n\n"
            f"说明:\n"
            f"  - 如果正/负值比例严重失衡 → 模型有偏置，但 roll_zscore 会自动修正\n"
            f"  - 如果标准差很小 → 预测值集中，需要标准化才能产生有效交易信号\n"
            f"  - 原始预测值量级小是正常的 (因为目标是收益率)",
            title="诊断结论"
        )

    def _display_model_diagnosis(self, model):
        """诊断模型参数 - 方案A"""
        logger.rule("模型参数诊断 (方案A)")

        diagnosis_rows = []

        # 检查 mean_head
        if hasattr(model, 'mean_head'):
            bias_val = model.mean_head.bias.item() if model.mean_head.bias is not None else None
            weight_mean = model.mean_head.weight.mean().item()
            weight_std = model.mean_head.weight.std().item()
            weight_min = model.mean_head.weight.min().item()
            weight_max = model.mean_head.weight.max().item()

            diagnosis_rows.extend([
                {'参数': 'mean_head.bias', '值': f"{bias_val:.6f}" if bias_val is not None else 'None'},
                {'参数': 'mean_head.weight.mean', '值': f"{weight_mean:.6f}"},
                {'参数': 'mean_head.weight.std', '值': f"{weight_std:.6f}"},
                {'参数': 'mean_head.weight.min', '值': f"{weight_min:.6f}"},
                {'参数': 'mean_head.weight.max', '值': f"{weight_max:.6f}"},
            ])

        # 检查 variance_head
        if hasattr(model, 'variance_head'):
            var_bias = model.variance_head.bias.item() if model.variance_head.bias is not None else None
            diagnosis_rows.append({
                '参数': 'variance_head.bias',
                '值': f"{var_bias:.6f}" if var_bias is not None else 'None'
            })

        if diagnosis_rows:
            df = pd.DataFrame(diagnosis_rows)
            logger.table(df, title="模型输出层参数")

        # 诊断结论
        if hasattr(model, 'mean_head') and model.mean_head.bias is not None:
            bias_val = model.mean_head.bias.item()

            # tanh 输出范围分析
            # tanh(bias) * 0.05 的效果
            import math
            tanh_bias_effect = math.tanh(bias_val) * 0.05

            if abs(bias_val) > 0.5:
                logger.panel(
                    f"⚠️ mean_head.bias = {bias_val:.6f}\n"
                    f"   tanh(bias) * 0.05 = {tanh_bias_effect:.6f}\n\n"
                    f"偏置值较大，可能是导致预测偏置的原因\n\n"
                    f"建议:\n"
                    f"  - 尝试方案B: 去除 bias 重新训练\n"
                    f"  - 尝试方案C: 推理时去均值",
                    title="诊断结论 - H2 可能成立"
                )
            elif abs(bias_val) > 0.1:
                logger.panel(
                    f"⚠️ mean_head.bias = {bias_val:.6f}\n"
                    f"   tanh(bias) * 0.05 = {tanh_bias_effect:.6f}\n\n"
                    f"偏置值中等，可能部分贡献预测偏置\n\n"
                    f"建议: 继续检查 last_hidden 分布 (H3)",
                    title="诊断结论 - H2 部分成立"
                )
            else:
                logger.panel(
                    f"✓ mean_head.bias = {bias_val:.6f} 接近0\n"
                    f"   tanh(bias) * 0.05 = {tanh_bias_effect:.6f}\n\n"
                    f"偏置可能来自 weight 或上游特征\n\n"
                    f"建议: 检查 last_hidden 分布 (H3)",
                    title="诊断结论 - H2 不成立，检查 H3"
                )