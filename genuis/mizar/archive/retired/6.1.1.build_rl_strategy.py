import os, copy, json
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix
from kdutils.macro2 import *
from lib.uvx import * 
from lib.rl001.signal import Config
from lib.rl001.train import train_model
from lib.rl001.predict import predict_test_set
from lib.rl001.analysis import analyze_run,batch_analyze_runs

def merge(min_data, daily_data):
    min_data['trade_time'] = pd.to_datetime(min_data['trade_time'])
    daily_data['trade_time'] = pd.to_datetime(daily_data['trade_time'])
    
    min_data['join_date'] = min_data['trade_time'].dt.normalize()
    daily_data['trade_time'] = daily_data['trade_time'].dt.normalize()
    merged_data = pd.merge(
        min_data,
        daily_data,
        left_on=['join_date', 'code'],   # 左表键
        right_on=['trade_time', 'code'], # 右表键
        how='left',                      # 保证分钟线行数不变
        suffixes=('', '_daily')          # 如果有重名列，右表加后缀
    )
    cols_to_drop = ['join_date']
    if 'trade_time_daily' in merged_data.columns:
        cols_to_drop.append('trade_time_daily')
    
    merged_data = merged_data.drop(columns=['trade_time_y'], errors='ignore') # 默认后缀是_x, _y
    merged_data = merged_data.drop(columns=['join_date'], errors='ignore')
    print(f"合并前行数: {len(min_data)}, 合并后行数: {len(merged_data)}")
    return merged_data

def load_data1(method, instruments, task_id, period, features, regime, ret_name):
    base_dirs = os.path.join(base_path, method, instruments, 'temp',
                             'model', str(task_id), str(period),
                               'rl', 'data')
    train_data = pd.read_feather(os.path.join(base_dirs,
                                              "train_data.feather"))
    
    val_data = pd.read_feather(os.path.join(base_dirs,
                                              "val_data.feather"))
    
    train_data.rename(columns={ret_name: "nxt1_ret"},
                      inplace=True)

    val_data.rename(columns={ret_name: "nxt1_ret"},
                    inplace=True)
    
    train_data = train_data[['trade_time','code', 'nxt1_ret'] + features + regime]
    val_data = val_data[['trade_time','code', 'nxt1_ret'] + features + regime]
    
    train_data = train_data.sort_values('trade_time').reset_index(drop=True)
    val_data = val_data.sort_values('trade_time').reset_index(drop=True)
    if train_data['code'].nunique() != 1:
        raise ValueError(f"train_data 不是单标的，检测到 {train_data['code'].nunique()} 个 code")
    if val_data['code'].nunique() != 1:
        raise ValueError(f"val_data 不是单标的，检测到 {val_data['code'].nunique()} 个 code")
    return train_data, val_data
    
    
    
def load_data2(method, instruments, task_id, period, features, regime, ret_name):
    base_dirs = os.path.join(base_path, method, instruments, 'temp',
                             'model', str(task_id), str(period),
                               'rl', 'data')
    test_data = pd.read_feather(os.path.join(base_dirs, "test_data.feather"))
    
    test_data.rename(columns={ret_name: "nxt1_ret"},
                      inplace=True)
    
    test_data = test_data[['trade_time','code', 'nxt1_ret'] + features + regime]
    test_data = test_data.sort_values('trade_time').reset_index(drop=True)
    if test_data['code'].nunique() != 1:
        raise ValueError(f"test_data 不是单标的，检测到 {test_data['code'].nunique()} 个 code")
    return test_data


def load_daily1(method, instruments, task_id, period, features):
    base_dirs = os.path.join(base_path, method, instruments, 'temp',
                             'model', str(task_id), str(period),
                               'rl', 'data')
    train_data = pd.read_feather(os.path.join(base_dirs,
                                              "train_regime.feather"))
    
    val_data = pd.read_feather(os.path.join(base_dirs,
                                              "val_regime.feather"))
    train_data = train_data[['trade_time','code'] + features]
    val_data = val_data[['trade_time','code'] + features]
    return train_data, val_data
    
def load_daily2(method, instruments, task_id, period, features):
    base_dirs = os.path.join(base_path, method, instruments, 'temp',
                             'model', str(task_id), str(period),
                               'rl', 'data')
    test_data = pd.read_feather(os.path.join(base_dirs,
                                              "test_regime.feather"))
    test_data = test_data[['trade_time','code'] + features]
    return test_data
    
    
def train(method, instruments, task_id, period, env_id, trade_id, model_id, train_id, feature_id, regime_id):
    file_dirs = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl')
    
    env_params, trade_params, model_params, train_params, selected_features, min_regime, daily_regime = load_rl_params(
        file_dirs=file_dirs, 
        trade_id=trade_id, model_id=model_id, feature_id=feature_id,
        env_id=env_id, train_id=train_id, regime_id=regime_id)
    
    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({'selected_features':selected_features})
    total_params.update({'min_regime':min_regime})
    total_params.update({'daily_regime':daily_regime})
    
    name = Params.create_tag(total_params)
    
    output_dir = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl', 'result',str(name))
    
    os.makedirs(output_dir, exist_ok=True)
    logger.configure(log_file=os.path.join(output_dir,f"model.log"))
    
    train_daily_data, val_daily_data = load_daily1(
        method=method, instruments=instruments, task_id=task_id, period=period, features=daily_regime)
    
    train_data, val_data = load_data1(method=method, instruments=instruments, period=period, 
                                      task_id=task_id, ret_name=trade_params['ret_name'],
                                      features=selected_features, regime=min_regime)
    pdb.set_trace()
    train_data = merge(min_data=train_data, daily_data=train_daily_data).dropna()
    val_data =  merge(min_data=val_data, daily_data=val_daily_data).dropna()
    
    env_config = {
        'mode': env_params['mode'],
        'holding_period': env_params['holding_period'],
        'max_pairs': env_params['max_pairs'],
        'max_allowed_position': env_params['max_allowed_position'],
        'use_cooldown': True,
        'cooldown_steps': env_params['cooldown_steps'],
        'masking_threshold_multiplier': env_params['masking_threshold_multiplier'],
        'episode_len': env_params['episode_len'],
        'reward_scale': env_params['reward_scale'],
        'cost_rate': trade_params['base_cost'], 
        'obs_noise_std': env_params['obs_noise_std'],
        #'debug_daily_logs': env_params.get('debug_daily_logs', True),
        'seed': 42
    }
    
    sac_config = {
        'learning_rate': model_params['learning_rate'],
        'buffer_size': model_params['buffer_size'],  # 示例使用较小缓冲区
        'learning_starts': model_params['learning_starts'],
        'batch_size':  model_params['batch_size'],
        'tau':  model_params['tau'],
        'gamma':  model_params['gamma'],
        'train_freq':  model_params['train_freq'],
        'gradient_steps':  model_params['gradient_steps'],
        'ent_coef':  model_params['ent_coef'],
        'target_update_interval':  model_params['target_update_interval'],
        'policy_kwargs': {
            'net_arch': {
                'pi': model_params['policy_kwargs']['net_arch']['pi'],
                'qf': model_params['policy_kwargs']['net_arch']['qf']
            }
        }
    }
    
    signal_config = Config(
        temperature=trade_params['temperature'],
        cash_score=trade_params['cash_score'],
        threshold_mode=trade_params['threshold_mode'],
        threshold=trade_params['threshold'],
        threshold_k=trade_params['threshold_k'],
        threshold_min=trade_params['threshold_min'],
        threshold_max=trade_params['threshold_max'],
        base_cost=trade_params['base_cost'],
        cost_multiplier=trade_params['cost_multiplier'],
        cost_mode=trade_params['cost_mode'],
        score_mapping=trade_params.get('score_mapping', 'conservative')
    )
    
    logger.info(f"  训练集: {len(train_data)} 行")
    logger.info(f"  校验集: {len(val_data)} 行")
    
    logger.info(f" env_config: {env_config}")
    logger.info(f" sac_config: {sac_config}")
    logger.info(f" signal_config: {signal_config}")
    logger.info(f" train_params: {train_params}")
    logger.info(f" env_params: {env_params}")
    logger.info(f" trade_params: {trade_params}")
    logger.info(f" model_params: {model_params}")
    logger.info(f" selected_features: {selected_features}")
    logger.info(f" min_regime: {min_regime}")
    logger.info(f" daily_regime: {daily_regime}")
    
    model, training_info = train_model(
            train_df=train_data,
            val_df=val_data,
            features=selected_features + min_regime + daily_regime,
            env_config=env_config,
            sac_config=sac_config,
            signal_config=signal_config,
            output_dir=output_dir,
            total_timesteps=train_params['total_timesteps'],
            eval_freq=train_params['eval_freq'],
            save_freq=train_params['save_freq'],
            verbose=1
        )
    
def predict(method, instruments, task_id, period, env_id, trade_id, model_id, train_id, feature_id,regime_id):
    file_dirs = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl')
    env_params, trade_params, model_params, train_params, selected_features, min_regime, daily_regime  = load_rl_params(
        file_dirs=file_dirs, 
        trade_id=trade_id, model_id=model_id, feature_id=feature_id,
        env_id=env_id, train_id=train_id, regime_id=regime_id)
    
    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({'selected_features':selected_features})
    total_params.update({'min_regime':min_regime})
    total_params.update({'daily_regime':daily_regime})
    
    name = Params.create_tag(total_params)
    
    test_daily_data = load_daily2(method=method, instruments=instruments,
                            task_id=task_id, period=period,
                            features=daily_regime
                            )
    test_data = load_data2(method=method, instruments=instruments, period=period, 
                                      task_id=task_id, ret_name=trade_params['ret_name'],
                                      regime=min_regime,
                                      features=selected_features)
    
    test_data = merge(min_data=test_data, daily_data=test_daily_data).dropna()
    pdb.set_trace()
    output_dir = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl', 'result',str(name))
    best_model_path = os.path.join(output_dir, "models", "best_model", "best_model")
    
    config_path = os.path.join(output_dir, "config.json")
    pdb.set_trace()
    signals_df = predict_test_set(
            model_path=best_model_path,
            config_path=config_path,
            test_df=test_data,
            output_path=os.path.join(output_dir, "metrics", "results.csv"),
            deterministic=True,
            return_details=True
        )
    
def analysis(method, instruments, task_id, period, env_id, trade_id, model_id, train_id, feature_id,regime_id,
             annual_trading_days=252, risk_free_rate=0.0):
    file_dirs = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl')
    env_params, trade_params, model_params, train_params, selected_features, min_regime, daily_regime = load_rl_params(
        file_dirs=file_dirs, 
        trade_id=trade_id, model_id=model_id, feature_id=feature_id,
        env_id=env_id, train_id=train_id, regime_id=regime_id)
    
    total_params = copy.deepcopy(trade_params)
    total_params.update(env_params)
    total_params.update(model_params)
    total_params.update(train_params)
    total_params.update({'selected_features':selected_features})
    total_params.update({'min_regime':min_regime})
    total_params.update({'daily_regime':daily_regime})
    
    name = Params.create_tag(total_params)
    
    output_dir = os.path.join(base_path, method, instruments, 'temp',
                               'model', str(task_id), str(period),
                               'rl', 'result', str(name))
    
    logger.configure(log_file=os.path.join(output_dir, f"model.log"))
    analysis_path = os.path.join(output_dir, "metrics", "analysis.json")
    report = analyze_run(
        run_dir=output_dir,
        output_path=analysis_path,
        annual_trading_days=annual_trading_days,
        risk_free_rate=risk_free_rate
    )
    
    perf = report.get("single_asset_performance", {})
    if perf:
        logger.info("单标的绩效摘要:")
        logger.info(f"  - 测试天数: {perf.get('test_days', 0)}")
        logger.info(f"  - 累计收益率: {perf.get('cumulative_return', 0.0):.2%}")
        logger.info(f"  - 年化收益率: {perf.get('annualized_return', 0.0):.2%}")
        logger.info(f"  - 年化波动率: {perf.get('annualized_volatility', 0.0):.2%}")
        logger.info(f"  - 夏普比率: {perf.get('sharpe_ratio', 0.0):.2f}")
        logger.info(f"  - 最大回撤: {perf.get('max_drawdown', 0.0):.2%}")
        logger.info(f"  - 日胜率: {perf.get('daily_win_rate', 0.0):.2%}")
        logger.info(f"  - 日盈亏比: {perf.get('daily_profit_loss_ratio', 0.0):.2f}")
        logger.info(f"  - 每bar平均换手率: {perf.get('avg_daily_turnover_per_bar', 0.0):.2%}")
        logger.info(f"  - 日累计换手率: {perf.get('avg_daily_turnover', 0.0):.2%}")
        logger.info(f"  - 累计换手率: {perf.get('total_turnover', 0.0):.2f}")
        logger.info(f"  - 日均交易成本: {perf.get('avg_daily_trade_cost', 0.0):.6f}")
        logger.info(f"  - 累计交易成本: {perf.get('total_trade_cost', 0.0):.6f}")
        logger.info(f"  - 日均持仓数量: {perf.get('avg_daily_holding_count', 0.0):.1f}")
        logger.info(f"  - 日均HHI(集中度): {perf.get('avg_daily_hhi', 0.0):.4f}")
    
    return report

if __name__ == '__main__':
    variant = Tactix().start()
    batch_analyze_runs('./records/cicso1/ims/temp/model/200037/15/rl/result/',
                       './records/cicso1/ims/temp/model/200037/15/rl/output')
    '''
    analysis(method=variant.method, 
          instruments=variant.instruments, 
          task_id=variant.task_id, 
          period=variant.period, 
          env_id=variant.env_id, 
          trade_id=variant.trade_id, 
          model_id=variant.model_id, 
          train_id=variant.train_id,
          feature_id=variant.feature_id,
          regime_id=variant.regime_id)
    '''
