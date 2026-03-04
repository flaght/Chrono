import json, os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

from kichaos.stable3.sac import SAC

from lib.rl002.signal import Config
from lib.rl002.envs import TradingEnv
from lib.rl002.features import CrossSectionalExtractor
from lib.rl002.custom_policy import CrossSectionalSACPolicy



class TradingSignalGenerator:
    """
    A股截面信号生成器
    
    将训练好的 SAC 模型转换为组合权重
    输出每只股票在每个时间步的目标权重
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 deterministic: bool = True):
        self.model_path = model_path
        self.config_path = config_path
        self.deterministic = deterministic
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        sac_config = self.config['sac_config']
        if 'policy_kwargs' in sac_config and isinstance(sac_config['policy_kwargs'], str):
            del sac_config['policy_kwargs']
        elif 'policy_kwargs' in sac_config and isinstance(sac_config['policy_kwargs'], dict):
            if 'features_extractor_class' in sac_config['policy_kwargs']:
                 del sac_config['policy_kwargs']
                 
        self.features = self.config['features']
        self.env_config = self.config['env_config']
        self.signal_config_dict = self.config['signal_config']
        
        # 构建配置
        self.signal_config = Config(
            min_weight=self.signal_config_dict['min_weight'],
            max_weight=self.signal_config_dict['max_weight'],
            normalize=self.signal_config_dict['normalize'],
            top_k=self.signal_config_dict['top_k'],
            cost_rate=self.signal_config_dict['cost_rate'],
            stamp_duty=self.signal_config_dict['stamp_duty'],
            turnover_penalty=self.signal_config_dict['turnover_penalty'],
            rebalance_window=self.signal_config_dict['rebalance_window'],
        )
        
        # 加载模型
        self.model = SAC.load(model_path)
        print(f"模型加载成功: {model_path}")
        
    def create_env(self, df: pd.DataFrame) -> TradingEnv:
        """创建环境"""
        env = TradingEnv(
            df=df,
            features=self.features,
            n_assets=self.env_config['n_assets'],
            episode_len=len(df['trade_time'].unique()) - 1,
            start_time=0,
            reward_scale=self.env_config['reward_scale'],
            signal_config=self.signal_config,
            strict_asset_alignment=self.env_config['strict_asset_alignment']
        )
        return env
    
    def predict_signals(self, 
                       df: pd.DataFrame,
                       start_time_index: Optional[int] = None,
                       return_details: bool = False) -> pd.DataFrame:
        """
        预测组合权重
        
        Returns:
            signals_df: 包含每个时间步的组合权重和收益信息
        """
        env = self.create_env(df)
        
        start_idx = start_time_index if start_time_index is not None else 0
        obs = env.reset(start_time_index=start_idx)
        
        results = []
        
        while True:
            # Record the time that current action/reward corresponds to.
            current_time_index = env.current_time_index
            action, _states = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, info = env.step(action)

            # Skip terminal padding step that returns empty info.
            if done:
                break
            
            # 获取时间
            t_idx = min(current_time_index, len(env.unique_times) - 1)
            trade_time = env.unique_times[t_idx] if t_idx < len(env.unique_times) else None
            
            result = {
                'trade_time': trade_time,
                'portfolio_return': info.get('portfolio_return', 0.0),
                'cost': info.get('cost', 0.0),
                'turnover': info.get('turnover', 0.0),
                'n_holdings': info.get('n_holdings', 0),
                'hhi': info.get('hhi', 0.0),
                'reward_raw': info.get('reward_raw', 0.0),
            }
            
            if return_details:
                result['top_weights'] = str(info.get('top_weights', []))
                result['total_turnover'] = info.get('total_turnover', 0.0)
                result['total_cost'] = info.get('total_cost', 0.0)
                result['total_portfolio_return'] = info.get('total_portfolio_return', 0.0)
            
            results.append(result)
        
        return pd.DataFrame(results)
        
         
    def predict_batch(self,
                     df: pd.DataFrame,
                     batch_size: int = 500,
                     overlap: int = 50) -> pd.DataFrame:
        """批量预测"""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= batch_size:
            raise ValueError(
                f"overlap must be smaller than batch_size to guarantee progress, got overlap={overlap}, batch_size={batch_size}"
            )

        all_results = []
        unique_times = sorted(df['trade_time'].unique())
        n_total = len(unique_times)
        
        start = 0
        while start < n_total:
            end = min(start + batch_size, n_total)
            batch_times = unique_times[start:end]
            batch_df = df[df['trade_time'].isin(batch_times)].copy()
            
            if len(batch_df) < 2:
                break
            
            batch_signals = self.predict_signals(batch_df, return_details=True)
            
            if start > 0 and not batch_signals.empty:
                batch_signals = batch_signals.iloc[overlap:]
            
            all_results.append(batch_signals)
            if end >= n_total:
                start = n_total
            else:
                next_start = end - overlap
                if next_start <= start:
                    raise RuntimeError(
                        f"batch prediction cannot make progress: start={start}, end={end}, overlap={overlap}"
                    )
                start = next_start
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()


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
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        signals_df.to_csv(output_path, index=False)
        print(f"预测结果已保存: {output_path}")
    
    return signals_df
