"""
A股截面选股评估工具

评估组合表现: 收益、夏普、回撤、换手率等
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .predict import predict_test_set, TradingSignalGenerator


@dataclass
class EvaluationMetrics:
    """评估指标"""
    total_return: float           # 总收益
    annualized_return: float      # 年化收益
    sharpe_ratio: float           # 夏普比率
    calmar_ratio: float           # 卡尔玛比率
    max_drawdown: float           # 最大回撤
    avg_turnover: float           # 平均换手率
    total_turnover: float         # 累计换手率
    total_cost: float             # 累计交易成本
    cost_return_ratio: float      # 成本收益比
    avg_n_holdings: float         # 平均持仓数量
    avg_hhi: float                # 平均持仓集中度
    n_rebalances: int             # 调仓次数
    win_rate: float               # 胜率 (日收益 > 0)
    profit_loss_ratio: float      # 盈亏比


def calculate_cumulative_returns(portfolio_returns: pd.Series) -> pd.Series:
    """计算累计收益"""
    return (1 + portfolio_returns).cumprod() - 1


def calculate_drawdown(cumulative_returns: pd.Series) -> Tuple[pd.Series, float]:
    """计算回撤"""
    wealth = 1 + cumulative_returns
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    return drawdown, max_drawdown


def evaluate_signals(signals_df: pd.DataFrame,
                    cost_rate: float = 0.0003,
                    steps_per_day: int = 240,
                    trading_days_per_year: int = 252) -> EvaluationMetrics:
    """
    评估组合表现
    
    Args:
        signals_df: predict 返回的 DataFrame, 需要包含 portfolio_return, cost, turnover 等
        cost_rate: 手续费率
        steps_per_day: 每天的时间步数
        trading_days_per_year: 年化交易天数
    """
    if len(signals_df) == 0:
        return EvaluationMetrics(
            total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            calmar_ratio=0.0, max_drawdown=0.0, avg_turnover=0.0,
            total_turnover=0.0, total_cost=0.0, cost_return_ratio=0.0,
            avg_n_holdings=0.0, avg_hhi=0.0, n_rebalances=0,
            win_rate=0.0, profit_loss_ratio=0.0
        )
    
    # 组合收益序列
    port_returns = signals_df['portfolio_return'].fillna(0)
    net_returns = port_returns - signals_df['cost'].fillna(0)
    
    # 累计收益
    cumulative = calculate_cumulative_returns(net_returns)
    total_return = float(cumulative.iloc[-1]) if len(cumulative) > 0 else 0.0
    
    # 年化
    n_steps = len(signals_df)
    n_years = n_steps / (steps_per_day * trading_days_per_year) if n_steps > 0 else 1.0
    annualized_return = (1 + total_return) ** (1 / max(n_years, 0.001)) - 1 if total_return > -1 else -1.0
    
    # 夏普
    if len(net_returns) > 1 and net_returns.std() > 0:
        sharpe_ratio = net_returns.mean() / net_returns.std() * np.sqrt(steps_per_day * trading_days_per_year)
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
    
    # 成本
    total_cost = float(signals_df['cost'].fillna(0).sum())
    cost_return_ratio = total_cost / max(abs(total_return), 1e-8)
    
    # 持仓统计
    avg_n_holdings = float(signals_df['n_holdings'].mean()) if 'n_holdings' in signals_df else 0.0
    avg_hhi = float(signals_df['hhi'].mean()) if 'hhi' in signals_df else 0.0
    
    # 调仓次数
    n_rebalances = int((turnovers > 0.001).sum())
    
    # 胜率
    win_rate = float((net_returns > 0).mean()) if len(net_returns) > 0 else 0.0
    
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
        cost_return_ratio=cost_return_ratio,
        avg_n_holdings=avg_n_holdings,
        avg_hhi=avg_hhi,
        n_rebalances=n_rebalances,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
    )


def print_evaluation_report(metrics: EvaluationMetrics):
    """打印评估报告"""
    print("=" * 50)
    print("A股截面选股策略评估报告")
    print("=" * 50)
    print(f"  总收益:         {metrics.total_return:.6f}")
    print(f"  年化收益:       {metrics.annualized_return:.4f}")
    print(f"  夏普比率:       {metrics.sharpe_ratio:.4f}")
    print(f"  卡尔玛比率:     {metrics.calmar_ratio:.4f}")
    print(f"  最大回撤:       {metrics.max_drawdown:.6f}")
    print("-" * 50)
    print(f"  平均换手率:     {metrics.avg_turnover:.6f}")
    print(f"  累计换手率:     {metrics.total_turnover:.4f}")
    print(f"  累计交易成本:   {metrics.total_cost:.6f}")
    print(f"  成本收益比:     {metrics.cost_return_ratio:.4f}")
    print("-" * 50)
    print(f"  平均持仓数量:   {metrics.avg_n_holdings:.1f}")
    print(f"  平均集中度(HHI): {metrics.avg_hhi:.4f}")
    print(f"  调仓次数:       {metrics.n_rebalances}")
    print(f"  胜率:           {metrics.win_rate:.4f}")
    print(f"  盈亏比:         {metrics.profit_loss_ratio:.4f}")
    print("=" * 50)


def evaluate_model(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    output_path: Optional[str] = None,
    deterministic: bool = True
) -> Tuple[pd.DataFrame, EvaluationMetrics]:
    """评估模型"""
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 预测
    signals_df = predict_test_set(
        model_path=model_path,
        config_path=config_path,
        test_df=test_df,
        deterministic=deterministic,
        return_details=True
    )
    
    # 评估
    signal_config = config.get('signal_config', {})
    metrics = evaluate_signals(
        signals_df=signals_df,
        cost_rate=signal_config.get('cost_rate', 0.0003),
    )
    
    print_evaluation_report(metrics)
    
    # 保存
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
