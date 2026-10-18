import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import json

from lib.rl001.predict import predict_test_set

@dataclass
class EvaluationMetrics:
    # 收益指标
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # 交易指标
    total_trades: int
    long_trades: int
    short_trades: int
    win_rate: float
    profit_loss_ratio: float
    avg_return_per_trade: float
    
    # 持仓指标
    avg_holding_period: float
    turnover: float
    avg_confidence: float
    
    # 成本指标
    total_cost: float
    cost_ratio: float  # 成本占总收益的比例



def calculate_returns(signals_df: pd.DataFrame, 
                      ret_col: str = 'ret_1min',
                      holding_period: int = 15) -> pd.Series:
    """
    计算累计收益
    
    Args:
        signals_df: 包含信号的 DataFrame，必须有 'signal' 和 ret_col 列
        ret_col: 收益率列名
        holding_period: 持仓周期
    
    Returns:
        cumulative_returns: 累计收益序列
    """
    # 计算每步收益：signal × ret_1min
    step_returns = signals_df['signal'] * signals_df[ret_col]
    
    # 计算累计收益
    cumulative_returns = (1 + step_returns).cumprod() - 1
    
    return cumulative_returns


def calculate_drawdown(cumulative_returns: pd.Series) -> Tuple[pd.Series, float]:
    """
    计算回撤
    
    Args:
        cumulative_returns: 累计收益序列
    
    Returns:
        drawdown: 回撤序列
        max_drawdown: 最大回撤
    """
    # 计算累计最大值
    cumulative_max = cumulative_returns.expanding().max()
    
    # 计算回撤
    drawdown = (cumulative_returns - cumulative_max) / (1 + cumulative_max)
    
    max_drawdown = drawdown.min()
    
    return drawdown, max_drawdown


def evaluate_signals(signals_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    cost_rate: float = 0.0001,
                    holding_period: int = 15,
                    trading_days_per_year: int = 252) -> EvaluationMetrics:
    """
    评估交易信号
    
    Args:
        signals_df: 预测信号 DataFrame（必须包含 'signal', 'direction', 'confidence', 'trade_time'）
        test_df: 测试数据 DataFrame（必须包含 'ret_1min' 和 'trade_time'）
        cost_rate: 单边手续费率
        holding_period: 持仓周期
        trading_days_per_year: 每年交易天数
    
    Returns:
        metrics: 评估指标
    """
    # 合并数据
    merged_df = pd.merge(
        signals_df[['trade_time', 'signal', 'direction', 'confidence', 'opened', 'expired_count']],
        test_df[['trade_time', 'ret_1min']],
        on='trade_time',
        how='inner'
    )
    
    if len(merged_df) == 0:
        raise ValueError("信号数据与测试数据无法匹配，请检查 trade_time 列")
    
    # 计算收益
    step_returns = merged_df['signal'] * merged_df['ret_1min']
    
    # 计算成本（开仓和平仓）
    # 假设每次开仓和平仓都产生成本
    costs = np.zeros(len(merged_df))
    if 'opened' in merged_df.columns:
        costs[merged_df['opened'] == True] += cost_rate
    if 'expired_count' in merged_df.columns:
        costs[merged_df['expired_count'] > 0] += merged_df.loc[merged_df['expired_count'] > 0, 'expired_count'] * cost_rate
    
    # 净收益
    net_returns = step_returns - costs
    
    # 累计收益
    cumulative_returns = (1 + net_returns).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0.0
    
    # 年化收益
    num_days = len(merged_df) / (trading_days_per_year * 240)  # 假设每天240分钟
    annualized_return = (1 + total_return) ** (1 / num_days) - 1 if num_days > 0 else 0.0
    
    # 回撤
    _, max_drawdown = calculate_drawdown(cumulative_returns)
    
    # 夏普比率
    if net_returns.std() > 0:
        sharpe_ratio = np.sqrt(252 * 240) * net_returns.mean() / net_returns.std()
    else:
        sharpe_ratio = 0.0
    
    # 卡玛比率
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    # 交易统计
    total_trades = (merged_df['direction'] != 0).sum()
    long_trades = (merged_df['direction'] == 1).sum()
    short_trades = (merged_df['direction'] == -1).sum()
    
    # 胜率（简化计算：收益为正的交易占比）
    trade_returns = []
    current_position = 0
    current_direction = 0
    entry_return = 0.0
    
    for i, row in merged_df.iterrows():
        if row['direction'] != 0 and current_direction == 0:
            # 开仓
            current_direction = row['direction']
            entry_return = 0.0
        elif current_direction != 0:
            # 持仓期间
            entry_return += row['signal'] * row['ret_1min']
            # 检查是否平仓
            if row['expired_count'] > 0 or (i == len(merged_df) - 1):
                # 平仓
                trade_returns.append(entry_return)
                current_direction = 0
                entry_return = 0.0
    
    if len(trade_returns) > 0:
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        profit_trades = [r for r in trade_returns if r > 0]
        loss_trades = [r for r in trade_returns if r < 0]
        if len(loss_trades) > 0:
            profit_loss_ratio = abs(np.mean(profit_trades) / np.mean(loss_trades)) if profit_trades else 0.0
        else:
            profit_loss_ratio = np.inf if profit_trades else 0.0
        avg_return_per_trade = np.mean(trade_returns)
    else:
        win_rate = 0.0
        profit_loss_ratio = 0.0
        avg_return_per_trade = 0.0
    
    # 持仓指标
    holding_periods = []
    current_holding_start = None
    for i, row in merged_df.iterrows():
        if row['direction'] != 0 and current_holding_start is None:
            current_holding_start = i
        elif row['direction'] == 0 and current_holding_start is not None:
            holding_periods.append(i - current_holding_start)
            current_holding_start = None
    
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
    
    # 换手率
    total_cost = costs.sum()
    turnover = total_cost / cost_rate if cost_rate > 0 else 0.0
    
    # 平均置信度
    avg_confidence = merged_df['confidence'].mean()
    
    # 成本比率
    gross_return = step_returns.sum()
    cost_ratio = total_cost / abs(gross_return) if gross_return != 0 else 0.0
    
    metrics = EvaluationMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        total_trades=total_trades,
        long_trades=long_trades,
        short_trades=short_trades,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        avg_return_per_trade=avg_return_per_trade,
        avg_holding_period=avg_holding_period,
        turnover=turnover,
        avg_confidence=avg_confidence,
        total_cost=total_cost,
        cost_ratio=cost_ratio
    )
    
    return metrics


def print_evaluation_report(metrics: EvaluationMetrics):
    """打印评估报告"""
    print("=" * 60)
    print("模型评估报告")
    print("=" * 60)
    
    print("\n【收益指标】")
    print(f"  总收益: {metrics.total_return:.4%}")
    print(f"  年化收益: {metrics.annualized_return:.4%}")
    print(f"  夏普比率: {metrics.sharpe_ratio:.4f}")
    print(f"  卡玛比率: {metrics.calmar_ratio:.4f}")
    print(f"  最大回撤: {metrics.max_drawdown:.4%}")
    
    print("\n【交易指标】")
    print(f"  总交易次数: {metrics.total_trades}")
    print(f"  多头交易: {metrics.long_trades}")
    print(f"  空头交易: {metrics.short_trades}")
    print(f"  胜率: {metrics.win_rate:.4%}")
    print(f"  盈亏比: {metrics.profit_loss_ratio:.4f}")
    print(f"  平均每笔收益: {metrics.avg_return_per_trade:.6f}")
    
    print("\n【持仓指标】")
    print(f"  平均持仓周期: {metrics.avg_holding_period:.2f} 分钟")
    print(f"  换手率: {metrics.turnover:.2f}")
    print(f"  平均置信度: {metrics.avg_confidence:.4f}")
    
    print("\n【成本指标】")
    print(f"  总成本: {metrics.total_cost:.6f}")
    print(f"  成本比率: {metrics.cost_ratio:.4%}")
    
    print("=" * 60)
    
def evaluate_model(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    output_path: Optional[str] = None,
    deterministic: bool = True
) -> Tuple[pd.DataFrame, EvaluationMetrics]:
    """
    评估模型（预测 + 评估）
    
    Args:
        model_path: 模型文件路径
        config_path: 配置文件路径
        test_df: 测试数据
        output_path: 输出文件路径（可选）
        deterministic: 是否使用确定性策略
    
    Returns:
        signals_df: 预测信号 DataFrame
        metrics: 评估指标
    """
    # 预测
    signals_df = predict_test_set(
        model_path=model_path,
        config_path=config_path,
        test_df=test_df,
        deterministic=deterministic,
        return_details=True
    )
    
    # 加载配置获取 cost_rate
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    cost_rate = config['signal_config'].get('base_cost', 0.0001)
    holding_period = config['env_config'].get('holding_period', 15)
    
    # 评估
    metrics = evaluate_signals(
        signals_df=signals_df,
        test_df=test_df,
        cost_rate=cost_rate,
        holding_period=holding_period
    )
    
    # 打印报告
    print_evaluation_report(metrics)
    
    # 保存评估结果
    if output_path:
        # 将 numpy 类型转换为内置类型以便 JSON 序列化
        def to_builtin(x):
            if isinstance(x, (np.generic,)):
                return x.item()
            return x
        
        metrics_dict = {
            'total_return': to_builtin(metrics.total_return),
            'annualized_return': to_builtin(metrics.annualized_return),
            'sharpe_ratio': to_builtin(metrics.sharpe_ratio),
            'calmar_ratio': to_builtin(metrics.calmar_ratio),
            'max_drawdown': to_builtin(metrics.max_drawdown),
            'total_trades': to_builtin(metrics.total_trades),
            'long_trades': to_builtin(metrics.long_trades),
            'short_trades': to_builtin(metrics.short_trades),
            'win_rate': to_builtin(metrics.win_rate),
            'profit_loss_ratio': to_builtin(metrics.profit_loss_ratio),
            'avg_return_per_trade': to_builtin(metrics.avg_return_per_trade),
            'avg_holding_period': to_builtin(metrics.avg_holding_period),
            'turnover': to_builtin(metrics.turnover),
            'avg_confidence': to_builtin(metrics.avg_confidence),
            'total_cost': to_builtin(metrics.total_cost),
            'cost_ratio': to_builtin(metrics.cost_ratio)
        }
        
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"\n评估结果已保存到: {output_path}")
    
    return signals_df, metrics
    