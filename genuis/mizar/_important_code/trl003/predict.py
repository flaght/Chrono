import json, os, pdb
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

from kichaos.stable3.sac import SAC

from lib.rl003.signal import Config
from lib.rl003.trade_env import TradingEnv


class TradingSignalGenerator:
    """
    期现正套信号生成器
    
    输出每个交易对在每个时间步的正套权重
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 deterministic: bool = True):
        self.model_path = model_path
        self.config_path = config_path
        self.deterministic = deterministic
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.features = self.config['features']
        self.env_config = self.config['env_config']
        self.signal_config_dict = self.config['signal_config']
        
        pdb.set_trace()
        self.signal_config = Config(
            max_weight=self.signal_config_dict.get('max_weight', 1.0),
            normalize=self.signal_config_dict.get('normalize', True),
            top_k=self.signal_config_dict.get('top_k', 0),
            spot_fee=self.signal_config_dict.get('spot_fee', 0.0001),
            futures_fee=self.signal_config_dict.get('futures_fee', 0.0002),
            min_basis_pct=self.signal_config_dict.get('min_basis_pct', 0.001),
            turnover_penalty=self.signal_config_dict.get('turnover_penalty', 0.0),
        )
        
        self.model = SAC.load(model_path)
        print(f"模型加载成功: {model_path}")
        
    def create_env(self, df: pd.DataFrame) -> TradingEnv:
        env = TradingEnv(
            df=df,
            features=self.features,
            n_pairs=self.env_config.get('n_pairs', 0),
            episode_len=len(df['trade_time'].unique()) - 1,
            start_time=0,
            reward_scale=self.env_config.get('reward_scale', 10000.0),
            signal_config=self.signal_config
        )
        return env
    
    def predict_signals(self, 
                       df: pd.DataFrame,
                       start_time_index: Optional[int] = None,
                       return_details: bool = False) -> pd.DataFrame:
        """预测正套权重"""
        env = self.create_env(df)
        
        start_idx = start_time_index if start_time_index is not None else 0
        obs = env.reset(start_time_index=start_idx)
        
        results = []
        done = False
        
        while not done:
            action, _states = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, info = env.step(action)
            
            t_idx = min(info.get('time_index', 0), len(env.unique_times) - 1)
            trade_time = env.unique_times[t_idx] if t_idx < len(env.unique_times) else None
            
            result = {
                'trade_time': trade_time,
                'arb_return': info.get('arb_return', 0.0),
                'funding_return': info.get('funding_return', 0.0),
                'cost': info.get('cost', 0.0),
                'turnover': info.get('turnover', 0.0),
                'n_holdings': info.get('n_holdings', 0),
                'hhi': info.get('hhi', 0.0),
                'weighted_basis': info.get('weighted_basis', 0.0),
                'avg_basis': info.get('avg_basis', 0.0),
                'reward_raw': info.get('reward_raw', 0.0),
            }
            
            if return_details:
                result['top_weights'] = str(info.get('top_weights', []))
                result['total_turnover'] = info.get('total_turnover', 0.0)
                result['total_cost'] = info.get('total_cost', 0.0)
                result['total_arb_return'] = info.get('total_arb_return', 0.0)
            
            results.append(result)
        
        return pd.DataFrame(results)
    
def predict_test_set(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    output_path: Optional[str] = None,
    deterministic: bool = True,
    return_details: bool = True
) -> pd.DataFrame:
    """预测测试集"""
    generator = TradingSignalGenerator(
        model_path=model_path,
        config_path=config_path,
        deterministic=deterministic
    )
    
    signals_df = generator.predict_signals(
        test_df, return_details=return_details
    )
    
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        signals_df.to_csv(output_path, index=False)
        print(f"预测结果已保存: {output_path}")
    
    return signals_df
