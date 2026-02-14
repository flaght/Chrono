import pdb
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from lib import logger

from lib.cux001 import FactorEvaluate1


class Evaluator(object):
    def __init__(
        self,
        fee: float = 0.0,
        resampling_win: int = 1,
        roll_win: int = 252,
        scale_method: str = "raw",
        scale:int=10000
    ):

        self.fee = fee
        self.resampling_win = max(1, int(resampling_win))
        self.roll_win = max(5, int(roll_win))
        self.scale_method = scale_method
        self.scale = scale

    
    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        mae = np.mean(np.abs(y_pred - y_true))
        return {"rmse": rmse, "mae": mae}
    
    def calculate_correlation_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        ic = np.corrcoef(y_pred, y_true)[0, 1]
        rank_ic, _ = spearmanr(y_pred, y_true)
        return {"ic": ic, "rank_ic": rank_ic}
    
    def calculate_direction_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        pred_direction = np.sign(y_pred)
        true_direction = np.sign(y_true)
        direction_acc = np.mean(pred_direction == true_direction)
        cm = confusion_matrix(true_direction, pred_direction, labels=[-1, 1])
        return direction_acc, cm
    
    def show_matrix(self, val_cm):
        logger.panel(
            f"(左上角, A)\n"
            f"含义: 在所有实际是下跌的样本中，模型正确地预测为下跌的数量\n"
            f"解读: 这是模型正确识别熊市的能力。这个数字越大，说明模型在市场下跌时能准确地发出预警，避免损失。这是好的预测。\n\n"
            f"(右下角, D)\n"
            f"含义: 在所有实际是上涨的样本中，模型正确地预测为上涨的数量。\n"
            f"解读: 这是模型正确识别牛市的能力。这个数字越大，说明模型能准确地抓住上涨机会，帮助盈利。这也是好的预测。。\n\n"
            f"(右上角, B)\n"
            f"含义: 在所有实际是下跌的样本中，模型错误地预测为上涨的数量。。\n"
            f" 这是非常危险的错误，被称为**“第一类错误 (Type I Error)”或“假正例 (False Positive)”。模型给了你一个“买入”的信号，但市场实际上却下跌了，这会导致直接的资金亏损**。。\n\n"
            f"(右下角, D)\n"
            f"含义: 在所有实际是上涨的样本中，模型错误地预测为下跌的数量。\n"
            f"解读: 这是一种**“机会成本”的错误，被称为“第二类错误 (Type II Error)”或“假负例 (False Negative)”**。市场实际上涨了，但模型却告诉你市场会下跌，让你错过了本可以盈利的机会。。。\n\n",
            title="混淆矩阵说明"
        )
        confusion_matrix_df = pd.DataFrame(val_cm, index=["实际下跌", "实际上涨"], columns= ["预测下跌", "预测上涨"])
        confusion_matrix_df.index.name = ""
        logger.table(confusion_matrix_df.reset_index(), title="混淆矩阵 (校验集)")

        total_pred = val_cm.sum()
        prob_table = pd.DataFrame(
                [
                    ("预测正确上涨", val_cm[1, 1], val_cm[1, 1] / total_pred),
                    ("预测正确下跌", val_cm[0, 0], val_cm[0, 0] / total_pred),
                    ("预测错误下跌", val_cm[1, 0], val_cm[1, 0] / total_pred),
                    ("预测错误上涨", val_cm[0, 1], val_cm[0, 1] / total_pred),
                ],
                columns=["类型", "样本数", "概率"],
            )
        logger.table(prob_table, title="方向预测概率表")

        actual_up_total = val_cm[1].sum()
        actual_down_total = val_cm[0].sum()
        cond_prob_rows = []
        if actual_up_total > 0:
            cond_prob_rows.append(
                (
                    "上涨记录",
                    actual_up_total,
                    "预测正确上涨",
                    val_cm[1, 1],
                    val_cm[1, 1] / actual_up_total,
                )
            )
            cond_prob_rows.append(
                (
                    "上涨记录",
                    actual_up_total,
                    "预测错误上涨",
                    val_cm[1, 0],
                    val_cm[1, 0] / actual_up_total,
                )
            )
        if actual_down_total > 0:
            cond_prob_rows.append(
                (
                    "下跌记录",
                    actual_down_total,
                    "预测正确下跌",
                    val_cm[0, 0],
                    val_cm[0, 0] / actual_down_total,
                )
            )
            cond_prob_rows.append(
                (
                    "下跌记录",
                    actual_down_total,
                    "预测错误下跌",
                    val_cm[0, 1],
                    val_cm[0, 1] / actual_down_total,
                )
            )

        cond_prob_table = pd.DataFrame(
                cond_prob_rows, columns=["样本集合","总样本数", "预测类型", "样本数", "条件概率"]
            )
        logger.table(cond_prob_table,title="条件概率表")


    ### 训练集 + 校验集评估模型
    def fitting_evaluate(self, y_train_true: np.ndarray,
        y_train_pred: np.ndarray,
        y_val_true: np.ndarray,
        y_val_pred: np.ndarray,
        dates_test: Optional[np.ndarray] = None)->Dict:

        logger.rule("模型评估绩效")

        # 将放大后的预测值缩小回原始尺度
        y_train_pred_scaled = y_train_pred / self.scale
        y_val_pred_scaled = y_val_pred / self.scale

        train_metrics = self.calculate_regression_metrics(y_train_true, y_train_pred_scaled)
        val_metrics = self.calculate_regression_metrics(y_val_true, y_val_pred_scaled)
        logger.panel(
            f"  RMSE 训练/测试: {train_metrics['rmse']:.6f} / {val_metrics['rmse']:.6f}\n"
            f"  MAE  训练/测试: {train_metrics['mae']:.6f} / {val_metrics['mae']:.6f}\n",
            title="基础回归指标"
        )

        train_corr = self.calculate_correlation_metrics(y_train_true, y_train_pred_scaled)
        test_corr = self.calculate_correlation_metrics(y_val_true, y_val_pred_scaled)
        logger.panel(
            f"  IC    训练/校验: {train_corr['ic']:.4f} / {test_corr['ic']:.4f}\n"
            f"  RankIC 训练/校验: {train_corr['rank_ic']:.4f} / {test_corr['rank_ic']:.4f}\n",
            title="相关性指标"
        )

        train_dir_acc, _ = self.calculate_direction_accuracy(y_train_true, y_train_pred_scaled)
        val_dir_acc, val_cm = self.calculate_direction_accuracy(y_val_true, y_val_pred_scaled)
        logger.panel(
            f"  训练集方向准确率: {train_dir_acc*100:.2f}% \n"
            f"  校验集方向准确率: {val_dir_acc*100:.2f}%",title="方向预测能力"
            )
        
        #self.show_matrix(val_cm=val_cm)



    ### 训练集 + 校验集 绩效评估
    def fitting_metrics(
            self, train_factors: pd.Series,
            val_factors: pd.Series,
            returns: pd.Series,
            period:int
    ):
        
        def metrics(factors, returns, name, period):
            data = pd.merge(factors, returns,
                              how='left',
                              left_index=True,
                              right_index=True)
            evaluate1 = FactorEvaluate1(factor_data=data.reset_index(),
                                factor_name='transformed',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=self.roll_win,
                                fee=0.000,
                                scale_method=self.scale_method,
                                expression="train",
                                resampling_win=self.resampling_win)
            stats1 = evaluate1.run()
            returns1 = evaluate1.cal_returns()
            stats1['name'] = name
            returns1['name'] = name
            return stats1,returns1
        
        stats_train,returns_train = metrics(factors=train_factors, returns=returns, name='train', period=period)
        stats_val,returns_val = metrics(factors=val_factors, returns=returns, name='val', period=period)
        stats_dt = pd.DataFrame([stats_train,stats_val]).set_index('name').reset_index()
        returns_dt = pd.DataFrame([returns_train,returns_val]).set_index('name').reset_index()
        logger.panel(
            f"resampling_win>{self.resampling_win}\n"
            f"roll_win>{self.roll_win}\n"
            f"scale_method>{self.scale_method}\n"
            f"period>{period}", "参数说明"
        )
        logger.table(stats_dt, "绩效说明")
        logger.table(returns_dt, "多空收益")

    def final_evaluate(self, y_test_true:np.ndarray,
                       y_test_pred:np.ndarray):
        # 将放大后的预测值缩小回原始尺度
        y_test_pred_scaled = y_test_pred / self.scale
        test_metrics = self.calculate_regression_metrics(y_test_true, y_test_pred_scaled)
        logger.panel(
            f"  RMSE 测试: {test_metrics['rmse']:.6f}\n"
            f"  MAE  测试: {test_metrics['mae']:.6f}\n",
            title="基础回归指标"
        )

        test_corr = self.calculate_correlation_metrics(y_test_true, y_test_pred_scaled)

        logger.panel(
            f"  IC    测试: {test_corr['ic']:.4f}\n"
            f"  RankIC 测试: {test_corr['rank_ic']:.4f}\n",
            title="相关性指标"
        )

        test_dir_acc, test_cm = self.calculate_direction_accuracy(y_test_true, y_test_pred_scaled)

        logger.panel(
            f"  测试集方向准确率: {test_dir_acc*100:.2f}%",title="方向预测能力"
            )
        
        #self.show_matrix(val_cm=test_cm)
        
    def final_metrics(self, test_factors: pd.Series,
            returns: pd.Series,
            period:int):
        def metrics(factors, returns, name, period):
            data = pd.merge(factors, returns,
                              how='left',
                              left_index=True,
                              right_index=True)
            evaluate1 = FactorEvaluate1(factor_data=data.reset_index(),
                                factor_name='transformed',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=self.roll_win,
                                fee=0.000,
                                scale_method=self.scale_method,
                                expression="test",
                                resampling_win=self.resampling_win)
            stats1 = evaluate1.run()
            returns1 = evaluate1.cal_returns()
            evaluate1.plot_results()
            evaluate1.save_results("./temp")
            stats1['name'] = name
            returns1['name'] = name
            return stats1,returns1
        
        stats_test,returns_test = metrics(factors=test_factors, returns=returns, name='test', period=period)
        stats_dt = pd.DataFrame([stats_test]).set_index('name').reset_index()
        returns_dt = pd.DataFrame([returns_test]).set_index('name').reset_index()

        logger.panel(
            f"resampling_win>{self.resampling_win}\n"
            f"roll_win>{self.roll_win}\n"
            f"scale_method>{self.scale_method}\n"
            f"period>{period}", "参数说明"
        )
        logger.table(stats_dt, "绩效说明")
        logger.table(returns_dt, "多空收益")
        return stats_test
