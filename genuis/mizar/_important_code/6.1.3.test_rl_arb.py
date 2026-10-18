"""
数字货币期现正套完整示例

正套 = 做多现货 + 做空期货
收益定义 (log return):
  log_return = log(S₂/S₁) - log(F₂/F₁)
  正值 = 正套盈利 (现货跑赢期货)
  
完整收益合成:
  y_basis = log(S₂/S₁) - log(F₂/F₁)     ← log return
  R_total = (exp(y_basis) - 1) + f - c    ← simple 口径合成
  y_total = log(1 + R_total)              ← 转回 log return
"""

import pandas as pd
import numpy as np

from lib.rl003.train import train_model
from lib.rl003.predict import predict_test_set, TradingSignalGenerator
from lib.rl003.evaluator import evaluate_model
from lib.rl003.signal import Config

def create_sample_data(n_times: int = 5000, n_pairs: int = 20) -> pd.DataFrame:
    """
    创建期现套利模拟数据
    
    模拟 N 个加密货币交易对:
      - log_return: 正套 log return = log(S₂/S₁) - log(F₂/F₁)
      - basis_pct: 基差率 (可选, 用于观测)
      - funding_rate: 资金费率
      - 因子特征: basis_deviation, basis_std, funding_ma, basis_momentum
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n_times, freq='1min')
    
    rows = []
    for pair_idx in range(n_pairs):
        pair_id = f'pair_{pair_idx:04d}'
        
        # 模拟现货和期货价格
        spot_price = np.zeros(n_times)
        futures_price = np.zeros(n_times)
        spot_price[0] = 100 + np.random.uniform(-10, 10)
        futures_price[0] = spot_price[0] * (1 + np.random.uniform(0.001, 0.005))
        
        for t in range(1, n_times):
            # 共同随机冲击
            common_shock = np.random.randn() * 0.001
            # 现货
            spot_price[t] = spot_price[t-1] * (1 + common_shock + np.random.randn() * 0.0003)
            # 期货 (带均值回归, 向现货收敛)
            basis_revert = -0.005 * (futures_price[t-1] / spot_price[t-1] - 1)
            futures_price[t] = futures_price[t-1] * (1 + common_shock + basis_revert + np.random.randn() * 0.0003)
            # 价格不能为负
            spot_price[t] = max(spot_price[t], 1.0)
            futures_price[t] = max(futures_price[t], 1.0)
        
        # 计算 log return (正套收益): log(S₂/S₁) - log(F₂/F₁)
        spot_log_ret = np.diff(np.log(spot_price), prepend=np.log(spot_price[0]))
        futures_log_ret = np.diff(np.log(futures_price), prepend=np.log(futures_price[0]))
        log_return = spot_log_ret - futures_log_ret  # 正值 = 正套盈利
        
        # 基差率 (用于观测)
        basis_pct = (futures_price - spot_price) / spot_price
        
        # 资金费率 (均摊到每分钟)
        funding_rate = np.random.uniform(1e-7, 1e-6, n_times)
        high_funding_mask = np.random.random(n_times) < 0.05
        funding_rate[high_funding_mask] *= 10
        
        # 因子特征
        window = 60
        basis_ma = pd.Series(basis_pct).rolling(window, min_periods=1).mean().values
        basis_deviation = basis_pct - basis_ma
        basis_std = pd.Series(basis_pct).rolling(window, min_periods=1).std().fillna(0).values
        funding_ma = pd.Series(funding_rate).rolling(window * 8, min_periods=1).mean().values
        short_ma = pd.Series(basis_pct).rolling(10, min_periods=1).mean().values
        long_ma = pd.Series(basis_pct).rolling(60, min_periods=1).mean().values
        basis_momentum = short_ma - long_ma
        
        for t in range(n_times):
            rows.append({
                'trade_time': dates[t],
                'pair_id': pair_id,
                'log_return': log_return[t],
                'basis_pct': basis_pct[t],
                'funding_rate': funding_rate[t],
                'basis_deviation': basis_deviation[t],
                'basis_std': basis_std[t],
                'funding_ma': funding_ma[t],
                'basis_momentum': basis_momentum[t],
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['trade_time', 'pair_id']).reset_index(drop=True)
    return df


def main():
    """主函数"""
    print("=" * 60)
    print("数字货币期现正套 SAC 模型 - 完整示例")
    print("=" * 60)
    print("正套 = 做多现货 + 做空期货")
    print("收益 = log(S₂/S₁) - log(F₂/F₁) + funding - cost")
    
    # ========== 1. 准备数据 ==========
    print("\n【步骤1】准备期现套利数据...")
    
    n_times = 5000
    n_pairs = 20
    
    all_data = create_sample_data(n_times, n_pairs)
    print(f"  总数据量: {len(all_data)} 行")
    print(f"  交易对数: {n_pairs}")
    print(f"  时间步数: {n_times}")
    
    print(f"  正套 log return 分布:")
    print(f"    均值: {all_data['log_return'].mean():.6f}")
    print(f"    标准差: {all_data['log_return'].std():.6f}")
    print(f"  基差率分布:")
    print(f"    均值: {all_data['basis_pct'].mean()*100:.4f}%")
    
    # 划分数据集
    unique_times = sorted(all_data['trade_time'].unique())
    train_size = int(n_times * 0.6)
    val_size = int(n_times * 0.2)
    
    train_times = unique_times[:train_size]
    val_times = unique_times[train_size:train_size+val_size]
    test_times = unique_times[train_size+val_size:]
    
    train_df = all_data[all_data['trade_time'].isin(train_times)].reset_index(drop=True)
    val_df = all_data[all_data['trade_time'].isin(val_times)].reset_index(drop=True)
    test_df = all_data[all_data['trade_time'].isin(test_times)].reset_index(drop=True)
    
    print(f"  训练集: {len(train_df)} 行")
    print(f"  校验集: {len(val_df)} 行")
    print(f"  测试集: {len(test_df)} 行")
    
    features = ['basis_deviation', 'basis_std', 'funding_ma', 'basis_momentum']
    print(f"  因子特征: {features}")
    
    # ========== 2. 配置参数 ==========
    print("\n【步骤2】配置期现套利参数...")
    
    env_config = {
        'n_pairs': n_pairs,
        'episode_len': 500,
        'reward_scale': 100000.0,
        'seed': 42,
    }
    
    sac_config = {
        'learning_rate': 3e-4,
        'buffer_size': 50000,
        'learning_starts': 500,
        'batch_size': 128,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'policy_kwargs': {
            'net_arch': {
                'pi': [256, 256],
                'qf': [256, 256]
            }
        }
    }
    
    signal_config = Config(
        max_weight=0.2,
        normalize=True,
        top_k=10,
        spot_fee=0.0001,
        futures_fee=0.0002,
        min_basis_pct=0.001,
        turnover_penalty=0.0,
    )
    
    print(f"  交易对数: {n_pairs}")
    print(f"  选对数 (top_k): {signal_config.top_k}")
    print(f"  单对成本: {signal_config.spot_fee + signal_config.futures_fee} (双边)")
    
    # ========== 3. 训练模型 ==========
    print("\n【步骤3】训练模型...")
    
    output_dir = './output/test007_arb_example'
    
    try:
        model, training_info = train_model(
            train_df=train_df,
            val_df=val_df,
            features=features,
            env_config=env_config,
            sac_config=sac_config,
            signal_config=signal_config,
            output_dir=output_dir,
            total_timesteps=10000,
            eval_freq=2000,
            save_freq=5000,
            verbose=1
        )
        print(f"  训练完成！最佳模型: {training_info['best_model_path']}")
        
    except Exception as e:
        print(f"  训练出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 4. 预测测试集 ==========
    print("\n【步骤4】预测测试集...")
    
    try:
        signals_df = predict_test_set(
            model_path=training_info['best_model_path'],
            config_path=training_info['config_path'],
            test_df=test_df,
            output_path='./output/test007_arb_predictions/signals.csv',
            deterministic=True,
            return_details=True
        )
        
        print(f"  预测完成，共 {len(signals_df)} 个时间步")
        print(f"  平均持仓对数: {signals_df['n_holdings'].mean():.1f}")
        print(f"  平均换手率: {signals_df['turnover'].mean():.6f}")
        
    except Exception as e:
        print(f"  预测出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 5. 评估模型 ==========
    print("\n【步骤5】评估模型...")
    
    try:
        signals_df, metrics = evaluate_model(
            model_path=training_info['best_model_path'],
            config_path=training_info['config_path'],
            test_df=test_df,
            output_path='./output/test007_arb_evaluation/metrics.json',
            deterministic=True
        )
    except Exception as e:
        print(f"  评估出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("\n核心设计要点：")
    print("  1. 方向固定: 做多现货 + 做空期货 (正套)")
    print("  2. 收益 = log(S₂/S₁) - log(F₂/F₁) + funding - cost")
    print("  3. 正值 = 盈利, 无需取负 (和 ArbMetrics 因子评估一致)")
    print("  4. 因子评估流程: ArbMetrics 找有效因子 → 放入 RL 训练")


if __name__ == '__main__':
    main()
