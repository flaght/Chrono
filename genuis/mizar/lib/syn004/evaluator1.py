"""
增强版评估模块（Evaluator1）

在原 evaluator 的基础上：
1. 保留基础回归指标、相关性指标、方向准确率、分位数分析等
2. 绩效指标参照 `lib/cux001.py` 中 `FactorEvaluate1` 的实现方式
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

from lib import logger
from lib.cux001 import FactorEvaluate1


@dataclass
class StrategyPerformance:
    total_ret: float
    avg_ret: float
    max_dd: float
    calmar: float
    sharpe_trade: float
    sharpe_ann: float
    win_rate: float
    profit_ratio: float
    turnover: float
    long_stats: Dict[str, float]
    short_stats: Dict[str, float]
    nav: np.ndarray


class Evaluator1:
    """
    评估器：保留 evaluator.py 的关键统计，同时对策略绩效部分
    引入 FactorEvaluate1 的统计方式
    """

    def __init__(
        self,
        fee: float = 0.0,
        annualization_factor: int = 252,
        resampling_win: int = 1,
        roll_win: int = 252,
        scale_method: str = "raw",
    ):
        self.fee = fee
        self.annualization_factor = annualization_factor
        self.resampling_win = max(1, int(resampling_win))
        self.roll_win = max(5, int(roll_win))
        self.scale_method = scale_method

    # ----------------- 基础指标 -----------------
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        mae = np.mean(np.abs(y_pred - y_true))
        return {"rmse": rmse, "mae": mae}

    @staticmethod
    def calculate_correlation_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        ic = np.corrcoef(y_pred, y_true)[0, 1]
        rank_ic, _ = spearmanr(y_pred, y_true)
        return {"ic": ic, "rank_ic": rank_ic}

    @staticmethod
    def calculate_direction_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        pred_direction = np.sign(y_pred)
        true_direction = np.sign(y_true)
        direction_acc = np.mean(pred_direction == true_direction)
        cm = confusion_matrix(true_direction, pred_direction, labels=[-1, 1])
        return direction_acc, cm

    @staticmethod
    def calculate_quantile_analysis(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        df = pd.DataFrame({"pred": y_pred, "actual": y_true})
        labels = [f"Q{i+1}" for i in range(n_quantiles)]
        if n_quantiles == 5:
            labels = ["Q1(最看空)", "Q2", "Q3", "Q4", "Q5(最看多)"]

        df["pred_quantile"] = pd.qcut(
            df["pred"],
            q=n_quantiles,
            labels=labels,
            duplicates="drop",
        )
        return df.groupby("pred_quantile")["actual"].agg(["mean", "std", "count"])

    # ----------------- 绩效指标（参照 FactorEvaluate1） -----------------
    @staticmethod
    def _default_trade_time(length: int) -> pd.DatetimeIndex:
        return pd.date_range(start="2000-01-01", periods=length, freq="T")

    def _prepare_factor_dataframe(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[np.ndarray],
    ) -> pd.DataFrame:
        if dates is not None:
            trade_time = pd.to_datetime(dates)
        else:
            trade_time = self._default_trade_time(len(y_true))

        df = pd.DataFrame(
            {
                "trade_time": trade_time,
                "factor": y_pred,
                "ret": y_true,
            }
        )
        return df

    @staticmethod
    def _side_stats_from_resample(df: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
        side = df.loc[mask, "net_ret"] if "net_ret" in df else pd.Series(dtype=float)
        if side.empty:
            return {"count": 0, "sum": 0.0, "mean": 0.0, "win_rate": 0.0}
        return {
            "count": int(side.count()),
            "sum": float(side.sum()),
            "mean": float(side.mean()),
            "win_rate": float((side > 0).mean()),
        }

    def calculate_strategy_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> StrategyPerformance:
        if len(y_true) == 0:
            return StrategyPerformance(
                total_ret=0.0,
                avg_ret=0.0,
                max_dd=0.0,
                calmar=np.nan,
                sharpe_trade=0.0,
                sharpe_ann=np.nan,
                win_rate=0.0,
                profit_ratio=np.inf,
                turnover=0.0,
                long_stats={"count": 0, "sum": 0.0, "mean": 0.0, "win_rate": 0.0},
                short_stats={"count": 0, "sum": 0.0, "mean": 0.0, "win_rate": 0.0},
                nav=np.array([]),
            )

        factor_df = self._prepare_factor_dataframe(y_true, y_pred, dates)
        adaptive_roll = min(self.roll_win, max(5, len(factor_df)))

        evaluator = FactorEvaluate1(
            factor_data=factor_df,
            resampling_win=self.resampling_win,
            factor_name="factor",
            ret_name="ret",
            roll_win=adaptive_roll,
            fee=self.fee,
            scale_method=self.scale_method,
            annualization_factor=self.annualization_factor,
        )
        stats = evaluator.run(is_check=False)
        resample_df = getattr(evaluator, "resample_data", pd.DataFrame())

        nav = resample_df["nav"].values if "nav" in resample_df else np.array([])
        sharpe_trade = stats.get("sharpe1", 0.0)
        sharpe_ann = stats.get("sharpe2", np.nan)

        if not resample_df.empty and "pos" in resample_df.columns:
            pos_series = resample_df["pos"]
            long_stats = self._side_stats_from_resample(resample_df, pos_series > 0)
            short_stats = self._side_stats_from_resample(resample_df, pos_series < 0)
        else:
            long_stats = {"count": 0, "sum": 0.0, "mean": 0.0, "win_rate": 0.0}
            short_stats = {"count": 0, "sum": 0.0, "mean": 0.0, "win_rate": 0.0}

        return StrategyPerformance(
            total_ret=float(stats.get("total_ret", 0.0)),
            avg_ret=float(stats.get("avg_ret", 0.0)),
            max_dd=float(stats.get("max_dd", 0.0)),
            calmar=float(stats.get("calmar", np.nan)),
            sharpe_trade=float(sharpe_trade),
            sharpe_ann=float(sharpe_ann),
            win_rate=float(stats.get("win_rate", 0.0)),
            profit_ratio=float(stats.get("profit_ratio", np.inf)),
            turnover=float(stats.get("turnover", 0.0)),
            long_stats=long_stats,
            short_stats=short_stats,
            nav=nav,
        )

    # ----------------- 主评估流程 -----------------
    def evaluate(
        self,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        dates_test: Optional[np.ndarray] = None,
    ) -> Dict:
        logger.print("\n" + "=" * 80)
        logger.print("Evaluator1：模型评估（参照 FactorEvaluate1 绩效框架）")
        logger.print("=" * 80)

        # 1. 基础回归指标
        logger.print("\n[1] 基础回归指标")
        logger.print("-" * 40)
        train_metrics = self.calculate_regression_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_regression_metrics(y_test, y_test_pred)
        logger.print(f"  RMSE 训练/测试: {train_metrics['rmse']:.6f} / {test_metrics['rmse']:.6f}")
        logger.print(f"  MAE  训练/测试: {train_metrics['mae']:.6f} / {test_metrics['mae']:.6f}")

        # 2. 相关性指标
        logger.print("\n[2] 相关性指标")
        logger.print("-" * 40)
        train_corr = self.calculate_correlation_metrics(y_train, y_train_pred)
        test_corr = self.calculate_correlation_metrics(y_test, y_test_pred)
        logger.print(
            f"  IC    训练/测试: {train_corr['ic']:.4f} / {test_corr['ic']:.4f}")
        logger.print(
            f"  RankIC 训练/测试: {train_corr['rank_ic']:.4f} / {test_corr['rank_ic']:.4f}")

        # 3. 方向准确率
        logger.print("\n[3] 方向预测能力")
        logger.print("-" * 40)
        train_dir_acc, _ = self.calculate_direction_accuracy(y_train, y_train_pred)
        test_dir_acc, test_cm = self.calculate_direction_accuracy(y_test, y_test_pred)
        logger.print(f"  训练集方向准确率: {train_dir_acc*100:.2f}%")
        logger.print(f"  测试集方向准确率: {test_dir_acc*100:.2f}%")
        logger.print("\n  混淆矩阵 (测试集)")
        logger.print("                预测下跌  预测上涨")
        logger.print(f"    实际下跌    {test_cm[0,0]:6d}    {test_cm[0,1]:6d}")
        logger.print(f"    实际上涨    {test_cm[1,0]:6d}    {test_cm[1,1]:6d}")

        # 概率表：正确/错误的上涨、下跌预测
        total_pred = test_cm.sum()
        if total_pred > 0:
            prob_table = pd.DataFrame(
                [
                    ("预测正确上涨", test_cm[1, 1], test_cm[1, 1] / total_pred),
                    ("预测正确下跌", test_cm[0, 0], test_cm[0, 0] / total_pred),
                    ("预测错误下跌", test_cm[1, 0], test_cm[1, 0] / total_pred),
                    ("预测错误上涨", test_cm[0, 1], test_cm[0, 1] / total_pred),
                ],
                columns=["类型", "样本数", "概率"],
            )
            logger.print("\n  方向预测概率表 (测试集)")
            logger.print(prob_table.to_string(index=False, formatters={"概率": "{:.4%}".format}))

        # 条件概率表：在实际方向条件下的预测准确性
        actual_up_total = test_cm[1].sum()
        actual_down_total = test_cm[0].sum()
        cond_prob_rows = []
        if actual_up_total > 0:
            cond_prob_rows.append(
                (
                    "上涨记录",
                    actual_up_total,
                    "预测正确上涨",
                    test_cm[1, 1],
                    test_cm[1, 1] / actual_up_total,
                )
            )
            cond_prob_rows.append(
                (
                    "上涨记录",
                    actual_up_total,
                    "预测错误上涨",
                    test_cm[1, 0],
                    test_cm[1, 0] / actual_up_total,
                )
            )
        if actual_down_total > 0:
            cond_prob_rows.append(
                (
                    "下跌记录",
                    actual_down_total,
                    "预测正确下跌",
                    test_cm[0, 0],
                    test_cm[0, 0] / actual_down_total,
                )
            )
            cond_prob_rows.append(
                (
                    "下跌记录",
                    actual_down_total,
                    "预测错误下跌",
                    test_cm[0, 1],
                    test_cm[0, 1] / actual_down_total,
                )
            )
        if cond_prob_rows:
            cond_prob_table = pd.DataFrame(
                cond_prob_rows,
                columns=["样本集合", "总样本数", "预测类型", "样本数", "条件概率"],
            )
            logger.print("\n  条件概率表 (按实际方向)")
            logger.print(
                cond_prob_table.to_string(
                    index=False, formatters={"条件概率": "{:.4%}".format}
                )
            )

        # 4. 策略绩效（FactorEvaluate1 风格）
        logger.print("\n[4] 策略绩效指标（FactorEvaluate1）")
        logger.print("-" * 40)
        train_perf = self.calculate_strategy_performance(y_train, y_train_pred)
        test_perf = self.calculate_strategy_performance(
            y_test,
            y_test_pred,
            dates=dates_test,
        )

        def _log_perf(label: str, perf: StrategyPerformance):
            logger.print(f"\n  [{label}]")
            logger.print(f"    累计收益 Total Return : {perf.total_ret:.6f}")
            logger.print(f"    平均收益 Avg Ret      : {perf.avg_ret:.6f}")
            logger.print(f"    最大回撤 Max DD      : {perf.max_dd:.4f}")
            logger.print(f"    Calmar Ratio         : {perf.calmar if not np.isnan(perf.calmar) else float('nan'):.4f}")
            logger.print(f"    Sharpe(逐笔)         : {perf.sharpe_trade:.4f}")
            logger.print(
                f"    Sharpe(年化)         : {perf.sharpe_ann if not np.isnan(perf.sharpe_ann) else float('nan'):.4f}"
            )
            logger.print(f"    胜率 Win Rate        : {perf.win_rate*100:.2f}%")
            logger.print(f"    盈亏比 Profit Ratio  : {perf.profit_ratio:.4f}")
            logger.print(f"    换手率 Turnover      : {perf.turnover:.4f}")

            logger.print("\n    [做多统计]")
            logger.print(
                f"      次数: {perf.long_stats['count']}, 累计收益: {perf.long_stats['sum']:.6f}, "
                f"平均收益: {perf.long_stats['mean']:.6f}, 胜率: {perf.long_stats['win_rate']*100:.2f}%"
            )
            logger.print("\n    [做空统计]")
            logger.print(
                f"      次数: {perf.short_stats['count']}, 累计收益: {perf.short_stats['sum']:.6f}, "
                f"平均收益: {perf.short_stats['mean']:.6f}, 胜率: {perf.short_stats['win_rate']*100:.2f}%"
            )

        _log_perf("训练集", train_perf)
        _log_perf("测试集", test_perf)

        # 5. 分位数分析
        logger.print("\n[5] 预测值分位数分析")
        logger.print("-" * 40)
        quantile_stats = self.calculate_quantile_analysis(y_test, y_test_pred)
        logger.print(quantile_stats)
        q5_q1_diff = quantile_stats["mean"].iloc[-1] - quantile_stats["mean"].iloc[0]
        logger.print(f"\n  Q5 vs Q1 平均收益差: {q5_q1_diff:.6f}")
        logger.print(
            "    ✓ 正向单调关系" if q5_q1_diff > 0 else "    ✗ 单调关系不成立"
        )

        return {
            "train": {
                **train_metrics,
                **train_corr,
                "direction_acc": train_dir_acc,
                "performance": train_perf,
            },
            "test": {
                **test_metrics,
                **test_corr,
                "direction_acc": test_dir_acc,
                "performance": test_perf,
                "confusion_matrix": test_cm,
                "quantile_stats": quantile_stats,
                "q5_q1_diff": q5_q1_diff,
            },
        }


