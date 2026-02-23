import pdb, traceback, os
import pandas as pd
import numpy as np
from lib.rl001.train import train_model
from lib.rl001.predict import predict_test_set
from lib.rl001.evaluator import evaluate_model
from lib.rl001.signal import Config

def create_sample_data(n_samples: int = 5000) -> pd.DataFrame:
    """创建示例数据"""
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    df = pd.DataFrame({
        'trade_time': dates,
        'ret_1min': np.random.randn(n_samples) * 0.001,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
    })
    
    return df

def train():
    all_data = create_sample_data(n_samples=50000)
    
    train_size = int(len(all_data) * 0.6)
    val_size = int(len(all_data) * 0.2)
    
    train_df = all_data.iloc[:train_size].copy()
    val_df = all_data.iloc[train_size:train_size+val_size].copy()
    test_df = all_data.iloc[train_size+val_size:].copy()
    
    features = ['feature1', 'feature2', 'feature3']
    
    env_config = {
        'mode': 'UNLOCK',
        'holding_period': 15,
        'max_allowed_position': 10,
        'use_cooldown': True,
        'cooldown_steps': 3,
        'include_market_features': False,  # 示例数据中没有市场特征
        'episode_len': 500,
        'reward_scale': 10000.0,
        'seed': 42
    }
    
    sac_config = {
        'learning_rate': 3e-4,
        'buffer_size': 50000,  # 示例使用较小缓冲区
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
                'pi': [128, 128],
                'qf': [128, 128]
            }
        }
    }
    
    signal_config = Config(
        threshold_mode='fixed',
        threshold=0.5,
        base_cost=0.0001,
        cost_multiplier=2000.0,
        cost_mode='fixed'
    )
    
    print("  配置完成")
    
    # ========== 3. 训练模型 ==========
    print("\n【步骤3】训练模型...")
    print("  注意：这是示例，实际训练需要更多步数")
    
    output_dir = './temp/rl/output/example_training'
    
    try:
        model, training_info = train_model(
            train_df=train_df,
            val_df=val_df,
            features=features,
            env_config=env_config,
            sac_config=sac_config,
            signal_config=signal_config,
            output_dir=output_dir,
            total_timesteps=10000,  # 示例使用较少步数
            eval_freq=2000,
            save_freq=5000,
            verbose=1
        )
        
        print(f"  训练完成！")
        print(f"  最佳模型: {training_info['best_model_path']}")
        print(f"  配置文件: {training_info['config_path']}")
        
    except Exception as e:
        traceback.print_exc()  
        print(f"  训练出错: {e}")
        print("  跳过训练步骤，使用已有模型进行演示")
        return
    

def predict():
    print("\n【步骤4】预测测试集...")
    output_dir = './temp/rl/output/example_training'
    all_data = create_sample_data(n_samples=50000)
    
    train_size = int(len(all_data) * 0.6)
    val_size = int(len(all_data) * 0.2)
    
    train_df = all_data.iloc[:train_size].copy()
    val_df = all_data.iloc[train_size:train_size+val_size].copy()
    test_df = all_data.iloc[train_size+val_size:].copy()
    best_model_path = os.path.join(output_dir, "models", "best_model", "best_model")
    config_path = os.path.join(output_dir, "config.json")
    
    try:
        signals_df = predict_test_set(
            model_path=best_model_path,
            config_path=config_path,
            test_df=test_df,
            output_path='./output/example_predictions/signals.csv',
            deterministic=True,
            return_details=True
        )
    except Exception as e:
        print(f"  预测出错: {e}")
        return
    
def evaluate():
    output_dir = './temp/rl/output/example_training'
    all_data = create_sample_data(n_samples=50000)
    
    train_size = int(len(all_data) * 0.6)
    val_size = int(len(all_data) * 0.2)
    
    train_df = all_data.iloc[:train_size].copy()
    val_df = all_data.iloc[train_size:train_size+val_size].copy()
    test_df = all_data.iloc[train_size+val_size:].copy()
    best_model_path = os.path.join(output_dir, "models", "best_model", "best_model")
    config_path = os.path.join(output_dir, "config.json")
    
    
    
    signals_df, metrics = evaluate_model(
            model_path=best_model_path,
            config_path=config_path,
            test_df=test_df,
            output_path='./output/example_evaluation/metrics.json',
            deterministic=True
        )
    pdb.set_trace()
    print('-->')
    
train()