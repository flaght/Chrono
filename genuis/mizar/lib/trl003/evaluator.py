import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from lib.rl003.predict import predict_test_set



@dataclass
class EvaluationMetrics:
    """评估指标"""
    total_return: float             # 总收益
    annualized_return: float        # 年化收益
    sharpe_ratio: float             # 夏普比率
    calmar_ratio: float             # 卡尔玛比率
    max_drawdown: float             # 最大回撤
    avg_turnover: float             # 平均换手率
    total_turnover: float           # 累计换手率
    total_cost: float               # 累计交易成本
    total_arb_return: float         # 累计套利收益 (不含成本)
    total_funding_return: float     # 累计资金费率收益
    cost_return_ratio: float        # 成本收益比
    avg_n_holdings: float           # 平均持仓对数
    avg_weighted_basis: float       # 平均加权基差
    win_rate: float                 # 胜率
    profit_loss_ratio: float        # 盈亏比


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1


def calculate_drawdown(cumulative: pd.Series) -> Tuple[pd.Series, float]:
    wealth = 1 + cumulative
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return drawdown, abs(drawdown.min())


def evaluate_signals(signals_df: pd.DataFrame,
                    steps_per_hour: int = 60,
                    trading_hours_per_year: int = 8760) -> EvaluationMetrics:
    """
    评估正套表现
    
    Args:
        signals_df: predict 返回的 DataFrame
        steps_per_hour: 每小时步数 (分钟数据 = 60)
        trading_hours_per_year: 年交易小时 (7*24*365/7 ≈ 8760, 数字货币全天交易)
    """
    if len(signals_df) == 0:
        return EvaluationMetrics(
            total_return=0, annualized_return=0, sharpe_ratio=0,
            calmar_ratio=0, max_drawdown=0, avg_turnover=0,
            total_turnover=0, total_cost=0, total_arb_return=0,
            total_funding_return=0, cost_return_ratio=0,
            avg_n_holdings=0, avg_weighted_basis=0,
            win_rate=0, profit_loss_ratio=0
        )
    
    # 净收益序列
    net_returns = signals_df['reward_raw'].fillna(0)
    
    # 累计
    cumulative = calculate_cumulative_returns(net_returns)
    total_return = float(cumulative.iloc[-1]) if len(cumulative) > 0 else 0.0
    
    # 年化
    n_steps = len(signals_df)
    steps_per_year = steps_per_hour * trading_hours_per_year
    n_years = n_steps / steps_per_year if n_steps > 0 else 1.0
    annualized_return = (1 + total_return) ** (1 / max(n_years, 0.001)) - 1 if total_return > -1 else -1.0
    
    # 夏普
    if len(net_returns) > 1 and net_returns.std() > 0:
        sharpe_ratio = net_returns.mean() / net_returns.std() * np.sqrt(steps_per_year)
    else:
        sharpe_ratio = 0.0
    
    # 回撤
    _, max_drawdown = calculate_drawdown(cumulative)
    
    # 卡尔玛
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
    
    # 换手率
    turnovers = signals_df['turnover'].fillna(0)
    avg_turnover = float(turnovers.mean())
    total_turnover = float(turnovers.sum())
    
    # 成本和收益
    total_cost = float(signals_df['cost'].fillna(0).sum())
    total_arb_return = float(signals_df['arb_return'].fillna(0).sum())
    total_funding_return = float(signals_df['funding_return'].fillna(0).sum()) if 'funding_return' in signals_df else 0.0
    gross_return = total_arb_return + total_funding_return
    cost_return_ratio = total_cost / max(abs(gross_return), 1e-8)
    
    # 持仓统计
    avg_n_holdings = float(signals_df['n_holdings'].mean()) if 'n_holdings' in signals_df else 0.0
    avg_weighted_basis = float(signals_df['weighted_basis'].mean()) if 'weighted_basis' in signals_df else 0.0
    
    # 胜率
    win_rate = float((net_returns > 0).mean())
    
    # 盈亏比
    winners = net_returns[net_returns > 0]
    losers = net_returns[net_returns < 0]
    avg_win = winners.mean() if len(winners) > 0 else 0.0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 1.0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    
    return EvaluationMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        avg_turnover=avg_turnover,
        total_turnover=total_turnover,
        total_cost=total_cost,
        total_arb_return=total_arb_return,
        total_funding_return=total_funding_return,
        cost_return_ratio=cost_return_ratio,
        avg_n_holdings=avg_n_holdings,
        avg_weighted_basis=avg_weighted_basis,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
    )


def print_evaluation_report(metrics: EvaluationMetrics):
    """打印评估报告"""
    print("=" * 55)
    print("数字货币期现正套策略评估报告")
    print("=" * 55)
    print(f"  总收益:           {metrics.total_return:.6f}")
    print(f"  年化收益:         {metrics.annualized_return:.4f}")
    print(f"  夏普比率:         {metrics.sharpe_ratio:.4f}")
    print(f"  卡尔玛比率:       {metrics.calmar_ratio:.4f}")
    print(f"  最大回撤:         {metrics.max_drawdown:.6f}")
    print("-" * 55)
    print(f"  套利收益(毛):     {metrics.total_arb_return:.6f}")
    print(f"  资金费率收益:     {metrics.total_funding_return:.6f}")
    print(f"  累计交易成本:     {metrics.total_cost:.6f}")
    print(f"  成本收益比:       {metrics.cost_return_ratio:.4f}")
    print("-" * 55)
    print(f"  平均换手率:       {metrics.avg_turnover:.6f}")
    print(f"  累计换手率:       {metrics.total_turnover:.4f}")
    print(f"  平均持仓对数:     {metrics.avg_n_holdings:.1f}")
    print(f"  平均加权基差:     {metrics.avg_weighted_basis:.6f}")
    print(f"  胜率:             {metrics.win_rate:.4f}")
    print(f"  盈亏比:           {metrics.profit_loss_ratio:.4f}")
    print("=" * 55)


def evaluate_model(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    output_path: Optional[str] = None,
    deterministic: bool = True
) -> Tuple[pd.DataFrame, EvaluationMetrics]:
    """评估模型"""
    signals_df = predict_test_set(
        model_path=model_path,
        config_path=config_path,
        test_df=test_df,
        deterministic=deterministic,
        return_details=True
    )
    metrics = evaluate_signals(signals_df)
    print_evaluation_report(metrics)
    
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        def to_builtin(x):
            if isinstance(x, (np.integer,)):
                return int(x)
            elif isinstance(x, (np.floating,)):
                return float(x)
            return x
        
        metrics_dict = {k: to_builtin(v) for k, v in metrics.__dict__.items()}
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"评估结果已保存: {output_path}")
    
    return signals_df, metrics

    