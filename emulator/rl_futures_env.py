# rl_futures_env.py
import gym, pdb
from gym import spaces
import numpy as np

from rl_engine_adapter import EngineAdapter
from rl_strategy_adapter import RLStrategyAdapter
from alphacopilot.api.calendars import makeSchedule

class FuturesRLEnv(gym.Env):
    def __init__(self, code, start_date, end_date, uri, lookback_window=30):
        super().__init__()
        pdb.set_trace()
        self.lookback_window = lookback_window
        self.action_space = spaces.Discrete(3)
        # 特征数(OHLCV=5) + 持仓状态(1)
        obs_shape = (lookback_window, 5 + 1) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        
        params = {'initial_capital': 1_000_000, 'multiplier': 200, 'commission_rate': {'open': 0.000023, 'close': 0.000023}}
        self.strategy = RLStrategyAdapter(strategy_id='rl_agent', params=params, lookback_window=lookback_window)
        self.adapter = EngineAdapter(code=code, uri=uri, rl_strategy=self.strategy)

        self.dates = makeSchedule(start_date, endDate=end_date, calendar='China.SSE', tenor='1b')
        self.date_iterator = iter(self.dates)
        self.current_bar = None

    def reset(self):
        try:
            current_date = next(self.date_iterator).strftime('%Y-%m-%d')
        except StopIteration:
            self.date_iterator = iter(self.dates)
            current_date = next(self.date_iterator).strftime('%Y-%m-%d')
            
        print(f"--- Resetting environment for date: {current_date} ---")
        
        # 重置适配器，这将返回第一个 bar 和结束标志
        bar, day_over = self.adapter.reset(current_date)
        
        # 填充 lookback_window
        # 循环 lookback_window 次来获取足够的初始历史数据
        for _ in range(self.lookback_window):
            if day_over:
                print(f"Warning: Not enough data on {current_date} to fill the lookback window.")
                break
            self.current_bar = bar
            bar, day_over = self.adapter.step()
        
        return self.strategy.get_observation()

    def step(self, action: int):
        pdb.set_trace()
        if self.current_bar is None:
            # 如果回合开始时就没有bar，或者已经结束
            obs = self.strategy.get_observation()
            return obs, 0.0, True, {'total_value': self.strategy.portfolio.net_value}

        # 1. Agent 执行动作
        self.strategy.execute_action(action, self.current_bar)
        
        # 2. 引擎推进一个时间步
        bar, day_over = self.adapter.step()
        self.current_bar = bar

        # 3. 获取结果
        obs = self.strategy.get_observation()
        reward = self.strategy.reward
        done = day_over
        info = {'total_value': self.strategy.portfolio.net_value}
        
        return obs, reward, done, info

    def render(self, mode='human'):
        if self.current_bar:
            print(f"Time: {self.current_bar.create_time}, "
                  f"Net Value: {self.strategy.portfolio.net_value:.2f}, "
                  f"Reward: {self.strategy.reward:.2f}")