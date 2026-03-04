import os, copy, json
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix
from kdutils.macro2 import base_path
from kdutils.logger import logger
from kdutils.macro2 import *

from lib.uvx import *
from lib.utils.params import Params
from lib.rl002.train import train_model
from lib.rl002.predict import predict_test_set
from lib.rl002.signal import Config
from lib.rl002.features import CrossSectionalExtractor
from lib.rl002.custom_policy import CrossSectionalSACPolicy

extractor_mapping = {
    'CrossSectionalExtractor':CrossSectionalExtractor
}


def load_data1(method, source, task_id, features, ret_name):
    pdb.set_trace()
    target_dir = os.path.join(base_path, method, source, 'rl', str(task_id))
    train_data = pd.read_feather(os.path.join(target_dir,
                                              "train_data.feather"))
    val_data = pd.read_feather(os.path.join(target_dir, "val_data.feather"))

    train_data.rename(columns={ret_name: "nxt1_ret"},
                      inplace=True)

    val_data.rename(columns={ret_name: "nxt1_ret"},
                    inplace=True)
    train_data = train_data[['trade_time','code', 'nxt1_ret'] + features]
    val_data = val_data[['trade_time','code', 'nxt1_ret'] + features]
    return train_data, val_data
    
def load_data2(method, source, task_id, features, ret_name):
    target_dir = os.path.join(base_path, method, source, 'rl', str(task_id))
    test_data = pd.read_feather(os.path.join(target_dir,
                                              "test_data.feather"))

    test_data.rename(columns={ret_name: "nxt1_ret"},
                      inplace=True)
    test_data = test_data[['trade_time','code', 'nxt1_ret'] + features]
    return test_data
    
    
def train(method, task_id, env_id, trade_id, model_id, train_id, feature_id):
    file_dirs = os.path.join(base_path, method, TASK_MAPPING[task_id]['source'], 
                             "temp", "trl", task_id)
    
   
    env_params, trade_params, model_params, train_params, selected_features = load_rl_params(
        file_dirs=file_dirs, 
        trade_id=trade_id, model_id=model_id, feature_id=feature_id,
        env_id=env_id, train_id=train_id)
    
    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({'selected_features':selected_features})
    
    name = Params.create_tag(total_params)
    
    os.makedirs(os.path.join(file_dirs, str(name)), exist_ok=True)
    logger.configure(log_file=os.path.join(file_dirs, str(name), f"model.log"))
    
    train_data, val_data = load_data1(method=method, source=TASK_MAPPING[task_id]['source'], 
                                      task_id=task_id, ret_name=trade_params['ret_name'],
                                      features=selected_features)
    
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
        'episode_len': env_params['episode_len'], 
        'reward_scale': env_params['reward_scale'],
        'strict_asset_alignment': env_params['strict_asset_alignment'],
        'seed': 42,
    }
    
    signal_config = Config(
        min_weight=0.0,             # 【A股专用】不能做空
        max_weight=trade_params['max_weight'],             # 单只股票最多买 10% (至少持仓 10 只)
        normalize=True,             # 强制选出的权重和为 100%
        top_k=trade_params['top_k'],                   # 从 5000 只里选出前 50 只让模型去分配资金 (选前 1%)
        cost_rate=0.0003,           # 佣金
        stamp_duty=0.0005,          # 印花税 (A股特有，仅卖出收)
        turnover_penalty=trade_params['turnover_penalty'],       # 换手惩罚
        rebalance_window=trade_params['rebalance_window'],
    )
    use_custom_policy = model_params['use_custom_policy']
    
    sac_config = {
        'learning_rate': float(model_params['learning_rate']),      
        'buffer_size': model_params['buffer_size'],        
        'learning_starts': model_params['learning_starts'],
        'batch_size': model_params['batch_size'],
        'tau': model_params['tau'],
        'gamma': model_params['gamma'],
        'train_freq': model_params['train_freq'],
        'gradient_steps': model_params['gradient_steps'],
        'ent_coef': model_params['ent_coef'],
        'target_update_interval': model_params['target_update_interval'],
    }
    
    if use_custom_policy:
        policy_class = CrossSectionalSACPolicy
        sac_config['policy_kwargs'] = {
            'n_assets': n_codes,
            'n_stock_features': len(features)
        }
    else:
        policy_class = 'MlpPolicy'
        policy_kwargs = model_params['policy_kwargs']
        extractor_kwargs = policy_kwargs['features_extractor_kwargs']
        net_arch = policy_kwargs['net_arch']
        
        sac_config['policy_kwargs'] = {
            'features_extractor_class': extractor_mapping[policy_kwargs['features_extractor_class']],
            'features_extractor_kwargs': dict(
                features_dim=extractor_kwargs['features_dim'], 
                n_assets=n_codes, 
                n_stock_features=len(features),
                stock_encoder_mid_dim=extractor_kwargs['encoder_mid_dim'],
                stock_encoder_out_dim=extractor_kwargs['encoder_out_dim'], 
            ),
            'net_arch': {
                'pi': net_arch['pi'],
                'qf': net_arch['qf']
            }
        }
    
    logger.info(f"  训练集: {len(train_data)} 行")
    logger.info(f"  校验集: {len(val_data)} 行")

    logger.info(f"  股票数量: {n_codes}")
    logger.info(f"  选股数 (top_k): {signal_config.top_k}")
    logger.info(f"  最大单股权重: {signal_config.max_weight}")
    logger.info(f"  佣金: {signal_config.cost_rate}")
    logger.info(f"  印花税: {signal_config.stamp_duty}")
    #logger.info(f"  网络结构: {sac_config['policy_kwargs']['net_arch']}")
    
    logger.info(f" env_config: {env_config}")
    logger.info(f" sac_config: {sac_config}")
    logger.info(f" signal_config: {signal_config}")
    logger.info(f" train_params: {train_params}")
    logger.info(f" env_params: {env_params}")
    logger.info(f" trade_params: {trade_params}")
    logger.info(f" model_params: {model_params}")
    logger.info(f" selected_features: {selected_features}")

    
    
    
    output_dir = os.path.join(base_path, method, TASK_MAPPING[task_id]['source'], 
                              "temp", "trl",  str(task_id), str(name))
    
    
    model, training_info = train_model(
            train_df=train_data,
            val_df=val_data,
            features=features,
            env_config=env_config,
            sac_config=sac_config,
            signal_config=signal_config,
            output_dir=output_dir,
            total_timesteps=train_params['total_timesteps'],
            eval_freq=train_params['eval_freq'],
            eval_n_episodes=train_params['eval_n_episodes'],
            save_freq=train_params['save_freq'],
            verbose=1,
            policy_class=policy_class
        )
        
    logger.info(f"  训练完成！")
    logger.info(f"  最佳模型: {training_info['best_model_path']}")
    config_path = os.path.join(output_dir, 'params.json')
    with open(config_path, 'w') as f:
        json.dump(total_params, f, indent=2, default=str)
    
  
def predict(method, task_id, env_id, trade_id, model_id, train_id, feature_id):
    file_dirs = os.path.join(base_path, method, TASK_MAPPING[task_id]['source'], 
                             "temp", "trl", task_id)
    env_params, trade_params, model_params, train_params, selected_features = load_rl_params(
        file_dirs=file_dirs, 
        trade_id=trade_id, model_id=model_id, feature_id=feature_id,
        env_id=env_id, train_id=train_id)
    
    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({'selected_features':selected_features})
    
    name = Params.create_tag(total_params)
    
    output_dir = os.path.join(base_path, method, TASK_MAPPING[task_id]['source'], 
                              "temp", "trl",  str(task_id), str(name))
    test_data = load_data2(method=method, source=TASK_MAPPING[task_id]['source'], 
                           task_id=task_id, ret_name=trade_params['ret_name'],
                           features=selected_features)
    best_model_path = os.path.join(output_dir, "models", "best_model", "best_model")
    config_path = os.path.join(output_dir, "training_config.json")
    signals_df = predict_test_set(
            model_path=best_model_path,
            config_path=config_path,
            test_df=test_data,
            output_path=os.path.join(output_dir, "metrics", "results.csv"),#'./temp/rl/output/test006_stock_example/signals.csv',
            deterministic=True,
            return_details=True
        )
    print(f"  预测完成，共 {len(signals_df)} 个时间步")
    print(f"  平均持仓数量: {signals_df['n_holdings'].mean():.1f}")
    print(f"  平均换手率: {signals_df['turnover'].mean():.6f}")
    print(f"  平均 HHI: {signals_df['hhi'].mean():.4f}")


if __name__ == '__main__':
    variant = Tactix().start()
    train(method=variant.method,
          task_id=variant.task_id,
          trade_id=variant.trade_id,
          env_id=variant.env_id,
          train_id=variant.train_id,
          model_id=variant.model_id,
          feature_id=variant.feature_id)
