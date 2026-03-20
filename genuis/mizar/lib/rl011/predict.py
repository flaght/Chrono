import json, os
from typing import Optional

import numpy as np
import pandas as pd

from kichaos.stable3.sac import SAC
from lib.rl011.envs import TradingEnv


class SignalGenerator:
    def __init__(self, model_path: str, config_path: str, deterministic: bool = True):
        self.model_path = model_path
        self.config_path = config_path
        self.deterministic = deterministic

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.model = SAC.load(model_path)
        self.features = self.config["features"]
        self.env_config = self.config["env_config"]
        self.signal_config = self.config.get("signal_config", {})
        
    def create_env(self, df: pd.DataFrame) -> TradingEnv:
        cfg = {
            "env_config": self.env_config,
            "signal_config": self.signal_config,
        }
        return TradingEnv(df=df, features=self.features, config=cfg)
    
    def predict_signals(self, df: pd.DataFrame, return_details: bool = True) -> pd.DataFrame:
        """
        输出每个时刻的 ER（权重）及环境内部累计结果。
        """
        env = self.create_env(df)
        obs = env.reset()
        results = []
        
        while True:
            current_step = env.current_step
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs_next, reward_scaled, done, _ = env.step(action)

            row_hist = env.history[-1] if env.history else {}
            trade_time = row_hist.get("trade_time", df.iloc[current_step].get("trade_time", current_step))
            action_raw = float(np.asarray(action).reshape(-1)[0])
            er_weight = float(row_hist.get("signal", action_raw))
            future_ret_h = float(row_hist.get("future_ret_h", action_raw))
            
            row = {
                "trade_time": trade_time,
                "er": er_weight,
                "action_raw": action_raw,
                "reward_scaled": float(reward_scaled),
                "future_ret_h": float(future_ret_h)
            }
            if return_details:
                row.update(
                    {
                        "direction": int(row_hist.get("direction", 0)),
                        "confidence": float(row_hist.get("confidence", abs(er_weight))),
                        "net_position": float(row_hist.get("net_position", er_weight)),
                        "reward_net_position": float(row_hist.get("reward_net_position", row_hist.get("net_position", er_weight))),
                        "active_signals": int(row_hist.get("active_signals", 0)),
                        "opened": bool(row_hist.get("opened", False)),
                        "current_ret": float(row_hist.get("current_ret", 0.0)),
                        "reward": float(row_hist.get("reward", reward_scaled)),
                        "trade_cost": float(row_hist.get("trade_cost", 0.0)),
                        "target_mode": str(row_hist.get("target_mode", "")),
                        "target_baseline": float(row_hist.get("target_baseline", 0.0)),
                        "target_ret_raw": float(row_hist.get("target_ret_raw", 0.0)),
                        "target_ret_excess": float(row_hist.get("target_ret_excess", 0.0)),
                        "target_ret": float(row_hist.get("target_ret", 0.0)),
                    }
                )

            results.append(row)
            obs = obs_next

            if done:
                break
        return pd.DataFrame(results)
            
            
        
def predict_test_set(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    output_path: Optional[str] = None,
    deterministic: bool = True,
    return_details: bool = True,
) -> pd.DataFrame:
    generator = SignalGenerator(
        model_path=model_path,
        config_path=config_path,
        deterministic=deterministic,
    )

    print(f"开始预测，测试集大小: {len(test_df)}")
    signals_df = generator.predict_signals(test_df, return_details=return_details)
    print(f"预测完成，生成 {len(signals_df)} 条记录")

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        signals_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"预测结果已保存到: {output_path}")

    return signals_df


# 向后兼容旧名称
ERSignalGenerator = SignalGenerator
