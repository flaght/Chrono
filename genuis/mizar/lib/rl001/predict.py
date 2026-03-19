import json, os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

from kichaos.stable3.sac import SAC

from lib.rl001.envs import TradingEnv
from lib.rl001.signal import Config
from lib.logger import logger

class TradingSignalGenerator:
    """
    交易信号生成器
    用于将训练好的 SAC 模型转换为交易信号
    可以直接用于交易系统或回测系统
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 deterministic: bool = True): # deterministic 是否使用确定性策略（预测时建议 True）
        self.model_path = model_path
        self.config_path = config_path
        self.deterministic = deterministic
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        # 重建信号配置
        sig_cfg = self.config['signal_config']
        self.signal_config = Config(
            temperature=sig_cfg['temperature'],
            cash_score=sig_cfg['cash_score'],
            threshold_mode=sig_cfg['threshold_mode'],
            threshold=sig_cfg['threshold'],
            threshold_k=sig_cfg['threshold_k'],
            threshold_min=sig_cfg['threshold_min'],
            threshold_max=sig_cfg['threshold_max'],
            base_cost=sig_cfg['base_cost'],
            cost_multiplier=sig_cfg['cost_multiplier'],
            cost_mode=sig_cfg['cost_mode'],
            score_mapping=sig_cfg.get('score_mapping', 'conservative')
        )
        
        
        
        # 加载模型
        self.model = SAC.load(model_path)
        
        self.features = self.config['features']
        self.env_config = self.config['env_config']
        
        
    def create_env(self, df: pd.DataFrame) -> TradingEnv:
        """创建环境（用于获取观测）"""
        env = TradingEnv(
            df=df,
            features=self.features,
            mode=self.env_config['mode'],
            holding_period=self.env_config['holding_period'],
            max_pairs=self.env_config['max_pairs'],
            max_allowed_position=self.env_config['max_allowed_position'],
            use_cooldown=self.env_config['use_cooldown'],
            cooldown_steps=self.env_config['cooldown_steps'],
            masking_threshold_multiplier=self.env_config['masking_threshold_multiplier'],
            episode_len=len(df),  # 预测时跑完整段
            seed=None,
            cost_rate=self.env_config['cost_rate'],
            reward_scale=self.env_config['reward_scale'],
            signal_config=self.signal_config,
            obs_noise_std=0.0,
        )
        return env
    
    def predict_signals(self, 
                       df: pd.DataFrame,
                       start_time_index: Optional[int] = None,
                       return_details: bool = False) -> pd.DataFrame:
        """
        预测交易信号
        
        Args:
            df: 测试数据（必须包含 features 和 trade_time）
                如果包含 nxt1_ret，将用于计算奖励；如果不包含，将自动填充为 0, 预测阶段不需要包含
            start_time_index: 起始时间索引（None 则从数据开始）
            return_details: 是否返回详细信息（包括 action, confidence 等）
        
        Returns:
            signals_df: 包含交易信号的 DataFrame
                - trade_time: 时间戳
                - signal: 交易信号 [-1, +1]
                - direction: 方向 {-1, 0, +1}
                - confidence: 置信度 [0, 1]
                - long_score: 多头置信度
                - short_score: 空头置信度
                - (如果 return_details=True) action, reward, net_position 等
        """
        # 复制数据以避免修改原始数据
        df = df.copy()
        
        if 'nxt1_ret' not in df.columns:
            df['nxt1_ret'] = 0.0
            print("警告: 数据中缺少 'nxt1_ret' 列，已自动填充为 0（预测模式）")
        elif df['nxt1_ret'].isna().any():
            df['nxt1_ret'] = df['nxt1_ret'].fillna(0.0)
            print("警告: 数据中 'nxt1_ret' 存在缺失值，已自动填充为 0（预测模式）")
        
        env = self.create_env(df)
        
        if start_time_index is not None:
            obs, info = env.reset(start_time_index=start_time_index)
        else:
            obs, info = env.reset()
            
        # 存储预测结果
        results = []
        # 遍历数据
        while not env.terminal:
            # 获取当前时间索引
            current_time_index = env.current_time_index
            
            # 模型预测（SAC 动作）
            action, _ = self.model.predict(
                obs, 
                deterministic=self.deterministic
            )
            
            # 执行一步（获取信号和详细信息）
            obs_next, reward, done, info = env.step(action)
            # 终止步允许 info 为空，避免错误截断
            if done and not info:
                break
            # 非终止步若信息缺失关键信息，则跳出避免 KeyError
            if 'signal' not in info or 'direction' not in info or 'confidence' not in info:
                break
            
            # 获取当前时间戳
            current_time = df.iloc[current_time_index]['trade_time']
            
            # 构建结果
            result = {
                'trade_time': current_time,
                'signal': info['signal'],
                'direction': info['direction'],
                'confidence': info['confidence'],
                'long_score': action[0],
                'short_score': action[1],
            }
            
            if return_details:
                result.update({
                    'action': action,
                    'reward': info['reward_raw'],
                    'reward_scaled': info['reward_scaled'],
                    'net_position': info['net_position'],
                    'active_signals': info['active_signals'],
                    'opened': info['opened'],
                    'expired_count': info['expired_count'],
                    'current_ret': info['current_ret'],
                    'trade_cost': info.get('trade_cost', 0.0),
                })
                if 'remaining_pairs' in info:
                    result['remaining_pairs'] = info['remaining_pairs']
            
            results.append(result)
            
            # 更新观测
            obs = obs_next
            
            if done:
                break
        
        # 转换为 DataFrame
        signals_df = pd.DataFrame(results)
        
        return signals_df
            
    def predict_batch(self,
                     df: pd.DataFrame,
                     batch_size: int = 500,
                     overlap: int = 50) -> pd.DataFrame:
        """
        批量预测（处理长序列数据）
        
        Args:
            df: 测试数据（必须包含 features 和 trade_time，nxt1_ret 可选）
            batch_size: 每批处理的步数
            overlap: 批次之间的重叠步数（用于平滑过渡）
        
        Returns:
            signals_df: 所有批次的预测结果
        """
        # 复制数据以避免修改原始数据
        df = df.copy()
        
        # 预测阶段：如果 nxt1_ret 缺失，自动填充为 0
        if 'nxt1_ret' not in df.columns:
            df['nxt1_ret'] = 0.0

        all_results = []
        # 计算批次
        total_steps = len(df)
        num_batches = (total_steps + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = max(0, i * batch_size - overlap)
            end_idx = min(total_steps, (i + 1) * batch_size)
            
            # 提取批次数据
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            # 预测
            batch_results = self.predict_signals(
                batch_df,
                start_time_index=0,
                return_details=True
            )
            
            # 如果是第一批，直接添加
            # 否则，去除重叠部分
            if i == 0:
                all_results.append(batch_results)
            else:
                # 去除前 overlap 行（重叠部分）
                batch_results = batch_results.iloc[overlap:]
                all_results.append(batch_results)
        
        
        # 合并所有结果
        if all_results:
            signals_df = pd.concat(all_results, ignore_index=True)
        else:
            signals_df = pd.DataFrame()
        
        return signals_df


def predict_test_set(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    output_path: Optional[str] = None,
    deterministic: bool = True,
    return_details: bool = True
): #预测结果
    """
        model_path: 模型文件路径
        config_path: 配置文件路径
        test_df: 测试数据（必须包含 features 和 trade_time，nxt1_ret 可选）
                如果 nxt1_ret 缺失，将自动填充为 0（预测模式）
        output_path: 输出文件路径（可选，保存为 CSV）
        deterministic: 是否使用确定性策略
        return_details: 是否返回详细信息
    """
    # 创建信号生成器
    generator = TradingSignalGenerator(
        model_path=model_path,
        config_path=config_path,
        deterministic=deterministic
    )
    
    # 预测
    logger.info(f"开始预测，测试集大小: {len(test_df)}")
    signals_df = generator.predict_signals(
        test_df,
        return_details=return_details
    )
    
    logger.info(f"预测完成，生成 {len(signals_df)} 条信号")
    logger.info(f"信号统计:")
    logger.info(f"  - 多头信号: {(signals_df['direction'] == 1).sum()}")
    logger.info(f"  - 空头信号: {(signals_df['direction'] == -1).sum()}")
    logger.info(f"  - 无信号: {(signals_df['direction'] == 0).sum()}")
    logger.info(f"  - 平均置信度: {signals_df['confidence'].mean():.4f}")
    
    # 保存结果
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        signals_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"预测结果已保存到: {output_path}")
    return signals_df
