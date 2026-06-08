import json, os, pdb
from typing import Optional
import numpy as np
import pandas as pd
from typing import Dict, List

from kichaos.stable3.sac import SAC
from lib.rl013.envs import TradingEnv


def _sanitize_dataframe(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    清洗训练数据中的 NaN/Inf，防止观测进入网络后产生 NaN 梯度。
    """
    out = df.copy()
    numeric_cols = list(features) + ["nxt1_ret"]
    existed_cols = [c for c in numeric_cols if c in out.columns]
    if not existed_cols:
        return out
    out[existed_cols] = out[existed_cols].apply(pd.to_numeric, errors="coerce")
    bad_mask = ~np.isfinite(out[existed_cols].to_numpy(dtype=np.float64))
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        print(f"[WARN] 检测到 {bad_count} 个非有限值(NaN/Inf)，已用 0.0 替换。")
    out[existed_cols] = out[existed_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


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
        df = _sanitize_dataframe(df, self.features)
        
        env_cfg = dict(self.env_config)
        env_cfg["mode"] = "test"   
        config = {
            "env_config": env_cfg,
            "signal_config": self.signal_config,
        }
        return TradingEnv(df=df, features=self.features, config=config)
    
    def predict_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        env = self.create_env(df)
        obs = env.reset()
        results = []
        while True:
            current_step = env.current_step
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            
            obs_next, reward_scaled, done, _ = env.step(action)
            
            row_hist = env.history[-1] if env.history else {}
            trade_time = row_hist.get("trade_time", df.iloc[current_step].get("trade_time", current_step))
        

            action_raw_str = row_hist.get("raw_action", str(action))
            action_soft_str = row_hist.get("soft_action", str(action))
            net_er_out = float(row_hist.get("net_er_out", 0.0))
            er_value = float(row_hist.get("er_value", 0.0))
            current_ret = float(row_hist.get("current_ret",0.0))
            future_ret_h = float(row_hist.get("future_ret_h", 0.0))
            
            row = {
                "trade_time": trade_time,
                "action_raw": action_raw_str,
                "action_soft":action_soft_str,
                "reward_scaled": float(reward_scaled),
                "future_ret_h": float(future_ret_h),
                "net_er_out": float(net_er_out),
                "er_value": float(er_value),
                "current_ret": float(current_ret)
            }
            
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
    deterministic: bool = True
) -> pd.DataFrame:
    generator = SignalGenerator(
        model_path=model_path,
        config_path=config_path,
        deterministic=deterministic,
    )
    print(f"开始预测，测试集大小: {len(test_df)}")
    signals_df = generator.predict_signals(test_df)
    print(f"预测完成，生成 {len(signals_df)} 条记录")
    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        os.makedirs(out_dir, exist_ok=True)
    print(output_path)
    signals_df.to_csv(output_path, index=False)