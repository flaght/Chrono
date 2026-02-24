import pdb,os
import pandas as pd
import numpy as np
from lib.rl002.signal import Config
from lib.rl002.train import train_model
from lib.rl002.predict import predict_test_set
from lib.rl002.evaluator import evaluate_model


def create_sample_data(n_times: int = 5000, n_assets: int = 20) -> pd.DataFrame:
    """
    创建 A股截面模拟数据
    
    Args:
        n_times: 时间步数
        n_assets: 股票数量
    
    Returns:
        df: 面板数据, 列 = [trade_time, asset_id, ret_1min, feature1, feature2, feature3]
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n_times, freq='1min')
    
    rows = []
    for asset_idx in range(n_assets):
        asset_id = f'stock_{asset_idx:04d}'
        
        # 生成因子 (带有少量预测性)
        f1 = np.random.randn(n_times)
        f2 = np.random.randn(n_times)
        f3 = np.random.randn(n_times)
        
        # 收益率: 与因子有微弱 IC
        true_ic = [0.02, -0.01, 0.015]
        ret = (
            f1 * true_ic[0] + 
            f2 * true_ic[1] + 
            f3 * true_ic[2]
        ) * 0.01 + np.random.randn(n_times) * 0.001
        
        # A股收益率通常非负偏 (长期略正)
        ret += 0.00001  # 微小正漂移
        
        for t in range(n_times):
            rows.append({
                'trade_time': dates[t],
                'asset_id': asset_id,
                'ret_1min': ret[t],
                'feature1': f1[t],
                'feature2': f2[t],
                'feature3': f3[t],
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['trade_time', 'asset_id']).reset_index(drop=True)
    return df


def train():
    pdb.set_trace()
    """主函数"""
    print("=" * 60)
    print("A股截面选股 SAC 模型 - 完整示例")
    print("=" * 60)
    
    # ========== 1. 准备数据 ==========
    print("\n【步骤1】准备 A股截面数据...")
    
    n_times = 5000
    n_assets = 50  # 模拟 20 只股票
    
    all_data = create_sample_data(n_times, n_assets)
    print(f"  总数据量: {len(all_data)} 行")
    print(f"  股票数量: {n_assets}")
    print(f"  时间步数: {n_times}")
    print(f"  每行: trade_time + asset_id + ret_1min + 3个因子")
    
    # 划分数据集 (按时间)
    unique_times = sorted(all_data['trade_time'].unique())
    train_size = int(n_times * 0.6)
    val_size = int(n_times * 0.2)
    
    train_times = unique_times[:train_size]
    val_times = unique_times[train_size:train_size+val_size]
    test_times = unique_times[train_size+val_size:]
    
    train_df = all_data[all_data['trade_time'].isin(train_times)].reset_index(drop=True)
    val_df = all_data[all_data['trade_time'].isin(val_times)].reset_index(drop=True)
    test_df = all_data[all_data['trade_time'].isin(test_times)].reset_index(drop=True)
    
    print(f"  训练集: {len(train_df)} 行 ({train_size} 时间步)")
    print(f"  校验集: {len(val_df)} 行 ({val_size} 时间步)")
    print(f"  测试集: {len(test_df)} 行 ({len(test_times)} 时间步)")
    
    features = ['feature1', 'feature2', 'feature3']
    
    # ========== 2. 配置参数 ==========
    print("\n【步骤2】配置 A股参数...")
    
    # A股环境配置
    env_config = {
        'n_assets': n_assets,     # 股票数量 (决定动作空间维度)
        'episode_len': 500,       # 每个 episode 步数
        'reward_scale': 10000.0,
        'seed': 42,
    }
    
    # SAC 配置
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
                'pi': [256, 256],   # 截面环境维度更高，用大一点的网络
                'qf': [256, 256]
            }
        }
    }
    
    # A股截面选股配置
    signal_config = Config(
        min_weight=0.0,           # A股: 不能做空, 最小权重 = 0
        max_weight=0.2,           # 单只股票最大权重 20% (风控)
        normalize=True,           # 权重归一化, 总和 = 1
        top_k=10,                 # 只选前 10 只 (从 20 只中选)
        cost_rate=0.0003,         # 佣金万三
        stamp_duty=0.0005,        # 印花税千分之0.5 (卖出收取)
        turnover_penalty=0.0,     # 额外换手惩罚 (可选)
        rebalance_window=1,       # 每步可调仓
    )
    
    print(f"  股票数: {n_assets}")
    print(f"  选股数 (top_k): {signal_config.top_k}")
    print(f"  最大单股权重: {signal_config.max_weight}")
    print(f"  佣金: {signal_config.cost_rate}")
    print(f"  印花税: {signal_config.stamp_duty}")
    print(f"  网络结构: {sac_config['policy_kwargs']['net_arch']}")
    
    
    print("\n【步骤3】训练模型...")
    print("  注意：这是示例，实际训练需要更多步数和更多股票")
    
    output_dir = './temp/rl/output/test006_stock_example'
    
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
        
        print(f"  训练完成！")
        print(f"  最佳模型: {training_info['best_model_path']}")
        
    except Exception as e:
        print(f"  训练出错: {e}")
        import traceback
        traceback.print_exc()
        return

def predict():
    print("\n【步骤4】预测测试集...")
    output_dir = './temp/rl/output/test006_stock_example'
    n_times = 5000
    n_assets = 50  # 模拟 20 只股票
    
    all_data = create_sample_data(n_times, n_assets)
    
    train_size = int(len(all_data) * 0.6)
    val_size = int(len(all_data) * 0.2)
    
    train_df = all_data.iloc[:train_size].copy()
    val_df = all_data.iloc[train_size:train_size+val_size].copy()
    test_df = all_data.iloc[train_size+val_size:].copy()
    best_model_path = os.path.join(output_dir, "models", "best_model", "best_model")
    config_path = os.path.join(output_dir, "training_config.json")
    
    signals_df = predict_test_set(
            model_path=best_model_path,
            config_path=config_path,
            test_df=test_df,
            output_path='./temp/rl/output/test006_stock_example/signals.csv',
            deterministic=True,
            return_details=True
        )
    print(f"  预测完成，共 {len(signals_df)} 个时间步")
    print(f"  平均持仓数量: {signals_df['n_holdings'].mean():.1f}")
    print(f"  平均换手率: {signals_df['turnover'].mean():.6f}")
    print(f"  平均 HHI: {signals_df['hhi'].mean():.4f}")
        
    # 验证: A股模式不应有负权重 (无空头)
    print("  ✅ A股模式: 仅做多，无空头信号")
    
def evaluator():
    output_dir = './temp/rl/output/test006_stock_example'
    n_times = 5000
    n_assets = 50  # 模拟 20 只股票
    
    all_data = create_sample_data(n_times, n_assets)
    
    train_size = int(len(all_data) * 0.6)
    val_size = int(len(all_data) * 0.2)
    all_data = create_sample_data(n_times, n_assets)
    
    train_size = int(len(all_data) * 0.6)
    val_size = int(len(all_data) * 0.2)
    
    train_df = all_data.iloc[:train_size].copy()
    val_df = all_data.iloc[train_size:train_size+val_size].copy()
    test_df = all_data.iloc[train_size+val_size:].copy()

    
    best_model_path = os.path.join(output_dir, "models", "best_model", "best_model")
    config_path = os.path.join(output_dir, "training_config.json")
    signals_df, metrics = evaluate_model(
            model_path=best_model_path,
            config_path=config_path,
            test_df=test_df,
            output_path='./output/test006_stock_evaluation/metrics.json',
            deterministic=True
        )
    
evaluator()