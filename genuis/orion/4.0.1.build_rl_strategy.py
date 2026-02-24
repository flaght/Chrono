import os, pdb
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

from kdutils.tactix import Tactix
from kdutils.macro2 import base_path
from kdutils.logger import logger

from lib.rl003.train import train_model
from lib.rl003.signal import Config


def load_data(method, period, source):
    target_dir = os.path.join(base_path, method, 'rl', period, source)
    train_data = pd.read_feather(os.path.join(target_dir,
                                              "train_data.feather"))
    val_data = pd.read_feather(os.path.join(target_dir, "val_data.feather"))

    train_data.rename(columns={"nxt1_ret_{0}h".format(1): "nxt1_ret"},
                      inplace=True)

    val_data.rename(columns={"nxt1_ret_{0}h".format(1): "nxt1_ret"},
                    inplace=True)
    return train_data, val_data


def train(method, period, source):
    pdb.set_trace()
    train_data, val_data = load_data(method=method,
                                     period=period,
                                     source=source)
    return_columns = train_data.filter(regex="^nxt1").columns.to_list()
    features = [
        f for f in train_data.columns
        if f not in ['trade_time', 'code'] + return_columns
    ]

    train_data = train_data[['trade_time', 'code', 'nxt1_ret'] + features]
    val_data = val_data[['trade_time', 'code', 'nxt1_ret'] + features]
    codes = train_data['code'].unique().tolist()
    n_pairs = len(codes)

    logger.info(f"  训练集: {len(train_data)} 行")
    logger.info(f"  校验集: {len(val_data)} 行")

    env_config = {
        'n_pairs': n_pairs,  # 等于 432
        'episode_len': 500,  # 保持 500 不错（大约跑完这1.5万天需要 30 个回合）
        'reward_scale':
        10000.0,  # 如果你的综合基础收益 (nxt1_ret) 通常在万分之一级别，乘 10万 是合理的，使奖励处于 [-10, 10] 之间
        'seed': 42,
    }

    sac_config = {
        'learning_rate': 1e-4,  # 略微调低 (3e-4 -> 1e-4)，应对更深层的网络和高维输入，防止过拟合
        'buffer_size': 50000,  # 这个非常合理。能存 5万步 回放池（约等于 3 遍完整训练集），有效防 OOM
        'learning_starts': 5000,  # 【重要】：调大！收集 10 个完整回合的随机数据后再开始学习，否则前期数据不足容易走偏
        'batch_size':
        512,  # 【重要】：调大！4700维输入，每次看 128 个样本太微观了（很容易看到全是震荡数据）。512 视野更广
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'ent_coef': 'auto',  # SAC 灵魂，自动调节探索率机制，保持
        'target_update_interval': 1,

        # 【最核心改动】：加深和放宽特征提取网络 (MlpPolicy)
        'policy_kwargs': {
            'net_arch': {
                # 输入层是 4700 维 -> 1024 -> 512 -> 256，漏斗型降维提取特征！
                # [256, 256] 绝对消化不了 4000多维度的特征。
                'pi': [1024, 512, 256],
                'qf': [1024, 512, 256]
            }
        }
    }

    signal_config = Config(
        max_weight=0.2,               # 【合理】：最大买 20%，这意味着模型最多同时重仓 5 个币（5*0.2 = 1.0）
        normalize=True,
        top_k=10,                     # 【极高过滤】：432个币只选 10 个（选前 2.3%）。非常好！模型需要做到极其自信。
        spot_fee=0.0001,
        futures_fee=0.0002,
        min_basis_pct=0.001,
        turnover_penalty=0.0,         # 前期探索先设 0，如果发现模型每步都在全换仓导致手续费过高，再设成 0.05 或 0.1 惩罚
    )

    logger.info(f"  交易对数: {n_pairs}")
    logger.info(f"  选对数 (top_k): {signal_config.top_k}")
    logger.info(
        f"  单对成本: {signal_config.spot_fee + signal_config.futures_fee} (双边)")

    output_dir = './output/test007_arb_example'

    model, training_info = train_model(train_df=train_data,
                                       val_df=val_data,
                                       features=features,
                                       env_config=env_config,
                                       sac_config=sac_config,
                                       signal_config=signal_config,
                                       output_dir=output_dir,
                                       total_timesteps=10000,
                                       eval_freq=2000,
                                       save_freq=5000,
                                       verbose=1)


if __name__ == '__main__':
    variant = Tactix().start()
    train(method=variant.method, period=variant.period, source=variant.source)
