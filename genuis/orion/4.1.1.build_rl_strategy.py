import os, pdb
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix
from kdutils.macro2 import base_path
from kdutils.logger import logger

from lib.rl002.train import train_model
from lib.rl002.signal import Config
from lib.rl002.features import CrossSectionalExtractor

def load_data(method, source, task_id, ret_name):
    target_dir = os.path.join(base_path, method, source, 'rl', str(task_id))
    train_data = pd.read_feather(os.path.join(target_dir,
                                              "train_data.feather"))
    val_data = pd.read_feather(os.path.join(target_dir, "val_data.feather"))

    train_data.rename(columns={ret_name: "nxt1_ret"},
                      inplace=True)

    val_data.rename(columns={ret_name: "nxt1_ret"},
                    inplace=True)
    return train_data, val_data
    
    
def train(method, source, task_id, ret_name):
    train_data, val_data = load_data(method=method, source=source, task_id=task_id, ret_name=ret_name)
    
    return_columns = train_data.filter(regex="^nxt1").columns.to_list() + train_data.filter(regex="^abret_").columns.to_list() 
    features = [
        f for f in train_data.columns
        if f not in ['trade_time', 'code'] + return_columns
    ]
    train_data = train_data[['trade_time', 'code', 'nxt1_ret'] + features]
    val_data = val_data[['trade_time', 'code', 'nxt1_ret'] + features]
    codes = train_data['code'].unique().tolist()
    n_codes = len(codes)
    
    env_config = {
        'n_assets': n_codes,
        'episode_len': 500,         # A股一年 ~252 个交易日，500步约等于 2 年重置一次状态
        'reward_scale': 10000.0,
        'seed': 42,
    }
    
    signal_config = Config(
        min_weight=0.0,             # 【A股专用】不能做空
        max_weight=0.1,             # 单只股票最多买 10% (至少持仓 10 只)
        normalize=True,             # 强制选出的权重和为 100%
        top_k=50,                   # 从 5000 只里选出前 50 只让模型去分配资金 (选前 1%)
        cost_rate=0.0003,           # 佣金
        stamp_duty=0.0005,          # 印花税 (A股特有，仅卖出收)
        turnover_penalty=0.0,       # 换手惩罚
        rebalance_window=1,
    )
    
    sac_config = {
        'learning_rate': 1e-4,      # 对高维输入降低学习率
        'buffer_size': 2000,        # 设成 2000 足够把整个训练集装进回放池 2 遍！
        'learning_starts': 100,     # 在这 100 步（天）内纯随机买入，积攒数据
        'batch_size': 128,          # 可以保持 128
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'policy_kwargs': {
            'features_extractor_class': CrossSectionalExtractor,
            
            # 向类的 __init__ 传参数
            'features_extractor_kwargs': dict(
                features_dim=256,        # 提取器最终吐出来的信号长度
                n_assets=n_codes, 
                n_stock_features=len(features)  # 你的原始因子数
            ),
            # 因为提取器已经输出了极高价值的 256 维浓缩信号
            # 所以 Pi 和 Qf 只需要两层小扁平网络做最终判断即可
            'net_arch': {
                'pi': [128, 128],
                'qf': [128, 128]
            }
        }
    }
    
    logger.info(f"  训练集: {len(train_data)} 行")
    logger.info(f"  校验集: {len(val_data)} 行")

    logger.info(f"  股票数量: {n_codes}")
    logger.info(f"  选股数 (top_k): {signal_config.top_k}")
    logger.info(f"  最大单股权重: {signal_config.max_weight}")
    logger.info(f"  佣金: {signal_config.cost_rate}")
    logger.info(f"  印花税: {signal_config.stamp_duty}")
    logger.info(f"  网络结构: {sac_config['policy_kwargs']['net_arch']}")
    
    output_dir = 'records/eicso0/ashare/temp/rl'
    
    model, training_info = train_model(
            train_df=train_data,
            val_df=val_data,
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
        
    logger.info(f"  训练完成！")
    logger.info(f"  最佳模型: {training_info['best_model_path']}")
    
    
    


if __name__ == '__main__':
    variant = Tactix().start()
    train(method=variant.method, source=variant.source, 
          task_id=variant.task_id, ret_name=variant.ret_name)