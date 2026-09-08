import json, os, pdb
from typing import Optional
import numpy as np
import pandas as pd
from typing import Dict, List

from kichaos.stable3.sac import SAC
from lib.rl015.envs import TradingEnv

def _sanitize_dataframe(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    清洗训练数据中的 NaN/Inf，防止观测进入网络后产生 NaN 梯度。
    """
    out = df.copy()
    # 只给输入特征补零；未来收益的缺失必须由环境识别为无效标签。
    numeric_cols = list(features)
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
        
        # 【多资产修改 14：预测尺度】读取训练保存的 ret_scale_by_code，
        # 保持各品种奖励尺度一致；模型输入形状也沿用保存的   时序模型配置。
        env_cfg = dict(self.env_config)
        env_cfg["mode"] = "test"   
        config = {
            "env_config": env_cfg,
            "signal_config": self.signal_config,
        }
        return TradingEnv(df=df, features=self.features, config=config)

    def predict_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        env = self.create_env(df)
        results = []
        # 【多资产修改 15：预测覆盖】原来遇到第一次 done 就停止；现在
        # 外层遍历所有品种，内层完成当前品种。一次调用不会只返回 RB。
        # 结果保留 code 区分相同时间的品种，输出按品种分块，不是全局时间排序。
        # 非训练环境每次 reset 顺序切换品种，因此循环全部品种 episode。
        for _ in range(env.n_assets):
            obs = env.reset()
            while True:
                current_step = env.current_step
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs_next, reward_scaled, done, _ = env.step(action)

                row_hist = env.history[-1] if env.history else {}
                trade_time = row_hist.get(
                    "trade_time", df.iloc[current_step].get("trade_time", current_step)
                )
                row = {
                    "trade_time": trade_time,
                    "code": row_hist.get("code", df.iloc[current_step].get("code", "")),
                    "label_valid": bool(row_hist["label_valid"]),
                    "confidence": float(row_hist["confidence"]),
                    "neutral_weight": float(row_hist["neutral_weight"]),
                    "is_neutral": bool(row_hist["is_neutral"]),
                    "action_raw": row_hist.get("raw_action", str(action)),
                    "action_soft": row_hist.get("soft_action", str(action)),
                    "reward_scaled": float(reward_scaled),
                    "future_ret_h": float(row_hist.get("future_ret_h", 0.0)),
                    "net_er_out": float(row_hist.get("net_er_out", 0.0)),
                    "er_value": float(row_hist.get("er_value", 0.0)),
                    "current_ret": float(row_hist.get("current_ret", 0.0)),
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
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        signals_df.to_csv(output_path, index=False)
        print(f"预测结果已保存到: {output_path}")
    # 不指定路径时只返回结果；指定路径时保存并返回同一份结果。
    return signals_df
