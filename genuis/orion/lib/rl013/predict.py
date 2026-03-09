import json, pdb
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch as th

from kichaos.stable3.sac import SAC

from lib.rl013.signal import (
    Config,
    calculate_portfolio_return,
    calculate_transaction_cost,
    calculate_turnover,
    scores_to_weights,
)


class TradingSignalGenerator:

    def __init__(
        self,
        model_path: str,
        config_path: str,
        deterministic: bool = True,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.deterministic = deterministic

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.features = self.config["features"]
        self.env_config = self.config["env_config"]
        self.use_custom_policy = self.config["use_custom_policy"]

        sig = self.config["signal_config"]
        self.signal_config = Config(
            min_weight=sig["min_weight"],
            max_weight=sig["max_weight"],
            normalize=sig["normalize"],
            top_k=sig["top_k"],
            cost_rate=sig["cost_rate"],
            turnover_penalty=sig["turnover_penalty"],
            rebalance_window=sig["rebalance_window"],
            softmax_temperature=sig["softmax_temperature"],
        )

        self.subset_size = self.env_config["subset_size"]
        self.n_features = len(self.features)

        self.model = SAC.load(model_path)

        if self.use_custom_policy:
            self.actor = self.model.policy.actor
            print(f"模型加载成功 (CrossSectionalSACPolicy): {model_path}")
        else:
            print(f"模型加载成功 (MlpPolicy): {model_path}")

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["trade_time", "code"] + self.features
        for col in required:
            if col not in df.columns:
                raise ValueError(f"missing required column: {col}")
        work = df.copy()
        if "nxt1_ret" not in work.columns:
            work["nxt1_ret"] = 0.0
        else:
            work["nxt1_ret"] = work["nxt1_ret"].fillna(0.0)
        return work.sort_values(["trade_time", "code"]).reset_index(drop=True)

    def _score_custom_policy(self, stock_features: np.ndarray,
                             portfolio_features: np.ndarray) -> np.ndarray:
        with th.no_grad():
            stocks_t = th.as_tensor(stock_features,
                                    device=self.model.device,
                                    dtype=th.float32)
            pf_t = th.as_tensor(portfolio_features,
                                device=self.model.device,
                                dtype=th.float32)
            scores = self.actor.score_assets(stocks_t,
                                             pf_t).detach().cpu().numpy()
        return scores.astype(np.float32)

    def _score_mlp_policy(self, stock_features: np.ndarray,
                          portfolio_features: np.ndarray) -> np.ndarray:
        n_stocks = stock_features.shape[0]
        all_scores = np.zeros(n_stocks, dtype=np.float32)

        for batch_start in range(0, n_stocks, self.subset_size):
            batch_end = min(batch_start + self.subset_size, n_stocks)
            batch_features = stock_features[batch_start:batch_end]

            # 零填充到 subset_size
            if batch_features.shape[0] < self.subset_size:
                pad = np.zeros(
                    (self.subset_size - batch_features.shape[0],
                     self.n_features),
                    dtype=np.float32,
                )
                batch_features = np.vstack([batch_features, pad])

            # 构造 obs: [features.flatten(), portfolio_features]
            obs = np.concatenate(
                [batch_features.flatten(),
                 portfolio_features]).astype(np.float32)
            obs = obs.reshape(1, -1)

            action, _ = self.model.predict(obs,
                                           deterministic=self.deterministic)
            action = action.flatten()

            # 取有效部分
            actual_count = batch_end - batch_start
            all_scores[batch_start:batch_end] = action[:actual_count]

        return all_scores

    def _score_cross_section(self, stock_features: np.ndarray,
                             portfolio_features: np.ndarray) -> np.ndarray:
        """统一入口：根据 policy 类型选择打分方式。"""
        if self.use_custom_policy:
            return self._score_custom_policy(stock_features,
                                             portfolio_features)
        else:
            return self._score_mlp_policy(stock_features, portfolio_features)

    def predict_signals(
        self,
        df: pd.DataFrame,
        top_k: Optional[int] = None,
        return_details: bool = False,
    ) -> pd.DataFrame:

        if top_k is not None and top_k < 0:
            raise ValueError(f"top_k_override must be >= 0, got {top_k}")

        work = self._prepare_data(df)
        grouped = {t: g for t, g in work.groupby("trade_time", sort=True)}
        unique_times = sorted(grouped.keys())

        asset_ids = sorted(work["code"].unique().tolist())
        code_to_index = {c: i for i, c in enumerate(asset_ids)}
        prev_weights = np.zeros(len(asset_ids), dtype=np.float32)
        last_turnover = 0.0

        total_turnover = 0.0
        total_cost = 0.0
        results = []
        pdb.set_trace()
        for step_idx, t in enumerate(unique_times):
            cs_data = grouped[t]
            codes = cs_data["code"].tolist()
            code_indices = np.array([code_to_index[c] for c in codes],
                                    dtype=np.int64)
            stock_features = cs_data[self.features].values.astype(np.float32)

            # Portfolio 状态特征
            hhi_prev = float(np.sum(prev_weights**2))
            holding_ratio_prev = float(
                np.sum(prev_weights > 1e-6) / max(len(prev_weights), 1))
            portfolio_features = np.array(
                [last_turnover, hhi_prev, holding_ratio_prev],
                dtype=np.float32)

            # 打分
            scores = self._score_cross_section(stock_features,
                                               portfolio_features)

            # 调仓判断
            should_rebalance = (self.signal_config.rebalance_window <= 1 or
                                (step_idx % self.signal_config.rebalance_window
                                 == 0))
            if should_rebalance:
                weights_top_k = top_k if top_k is not None else 0
                subset_weights = scores_to_weights(scores, self.signal_config,
                                                   weights_top_k)
                new_weights = np.zeros_like(prev_weights)
                new_weights[code_indices] = subset_weights
            else:
                new_weights = prev_weights.copy()
                subset_weights = new_weights[code_indices]

            # 指标
            turnover = calculate_turnover(prev_weights, new_weights)
            cost = calculate_transaction_cost(prev_weights, new_weights,
                                              self.signal_config)
            returns = cs_data["nxt1_ret"].values.astype(np.float32)
            portfolio_return = calculate_portfolio_return(
                subset_weights, returns)

            total_turnover += turnover
            total_cost += cost
            last_turnover = float(turnover)

            net_portfolio_return = portfolio_return - cost
            
            n_holdings = int(np.sum(subset_weights > 1e-8))
            hhi = float(np.sum(subset_weights**2))

            row = {
                "trade_time": t,
                "portfolio_return": float(portfolio_return),
                "cost": float(cost),
                "turnover": float(turnover),
                "n_holdings": n_holdings,
                "hhi": hhi,
                "net_portfolio_return": float(net_portfolio_return),
                "rebalanced": bool(should_rebalance),
                "top_k_used": int(top_k) if top_k is not None else 0,
            }
            if return_details:
                top_weights = np.sort(subset_weights)[::-1][:5]
                row["top_weights"] = str(top_weights.tolist())
                row["total_turnover"] = float(total_turnover)
                row["total_cost"] = float(total_cost)
            results.append(row)

            # 权重漂移
            drifted = subset_weights * (1.0 + returns)
            if np.sum(drifted) > 0:
                drifted = drifted / np.sum(drifted)
            else:
                drifted = np.zeros_like(drifted)
            next_prev = np.zeros_like(prev_weights)
            next_prev[code_indices] = drifted
            prev_weights = next_prev

        return pd.DataFrame(results)


def predict_test_set(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    top_k: Optional[int] = None,
    output_path: Optional[str] = None,
    deterministic: bool = True,
    return_details: bool = True,
) -> pd.DataFrame:
    generator = TradingSignalGenerator(
        model_path=model_path,
        config_path=config_path,
        deterministic=deterministic,
    )
    
    signals_df = generator.predict_signals(df=test_df,
                                           top_k=top_k,
                                           return_details=return_details)

    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        signals_df.to_csv(output_path, index=False)
        print(f"预测结果已保存: {output_path}")
    return signals_df
