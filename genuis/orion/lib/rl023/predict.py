import json, os, pdb
from typing import Optional

import numpy as np
import pandas as pd
import torch as th

from lib.cms003.metrics import Metrics
from kichaos.stable3.sac import SAC

from lib.rl023.signal import (
    Config,
    calculate_portfolio_return,
    calculate_transaction_cost,
    calculate_turnover,
    rank_ic,
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

        self.sampling_mode = self.env_config["sampling_mode"]
        self.action_mode = self.env_config["action_mode"]
        self.output_mode = self.env_config["output_mode"]
        self.include_portfolio_state = bool(
            self.env_config.get("include_portfolio_state",
                                self.sampling_mode == "sequential"))
        if self.sampling_mode == "random":
            self.include_portfolio_state = False

        if self.action_mode == "weights" and self.sampling_mode != "sequential":
            raise ValueError(
                "逻辑冲突: `action_mode='weights'` 必须搭配 `sampling_mode='sequential'`。\n"
                "推断时基于历史轨迹的持仓继承(turnover)必须依赖时间序列的连续性。")

        if self.sampling_mode == "random" and self.action_mode != "raw_ic":
            raise ValueError(
                "逻辑冲突: `sampling_mode='random'` 必须搭配 `action_mode='raw_ic'`。")

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
        self.stock_scores_df: Optional[pd.DataFrame] = None

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

    def _build_portfolio_features(
        self,
        prev_weights: np.ndarray,
        last_turnover: float,
    ) -> np.ndarray:
        if not self.include_portfolio_state:
            return np.zeros((0, ), dtype=np.float32)

        hhi_prev = float(np.sum(prev_weights**2))
        holding_ratio_prev = float(
            np.sum(prev_weights > 1e-6) / max(len(prev_weights), 1))
        return np.array([last_turnover, hhi_prev, holding_ratio_prev],
                        dtype=np.float32)

    def _score_custom_policy(
        self,
        stock_features: np.ndarray,
        portfolio_features: np.ndarray,
    ) -> np.ndarray:
        activation = "tanh" if self.action_mode == "raw_ic" else "sigmoid"
        with th.no_grad():
            stocks_t = th.as_tensor(stock_features,
                                    device=self.model.device,
                                    dtype=th.float32)
            pf_t = None
            if self.include_portfolio_state:
                pf_t = th.as_tensor(portfolio_features,
                                    device=self.model.device,
                                    dtype=th.float32)
            scores = self.actor.score_assets(
                stocks_t,
                pf_t,
                output_activation=activation,
            ).detach().cpu().numpy()

        return scores.astype(np.float32)

    def _score_mlp_policy(
        self,
        stock_features: np.ndarray,
        portfolio_features: np.ndarray,
    ) -> np.ndarray:
        n_stocks = stock_features.shape[0]
        all_scores = np.zeros(n_stocks, dtype=np.float32)

        for batch_start in range(0, n_stocks, self.subset_size):
            batch_end = min(batch_start + self.subset_size, n_stocks)
            batch_features = stock_features[batch_start:batch_end]

            if batch_features.shape[0] < self.subset_size:
                pad = np.zeros(
                    (self.subset_size - batch_features.shape[0],
                     self.n_features),
                    dtype=np.float32,
                )
                batch_features = np.vstack([batch_features, pad])

            obs_parts = [batch_features.flatten()]
            if self.include_portfolio_state:
                obs_parts.append(portfolio_features)
            obs = np.concatenate(obs_parts).astype(np.float32).reshape(1, -1)

            action, _ = self.model.predict(obs,
                                           deterministic=self.deterministic)
            action = action.flatten()

            actual_count = batch_end - batch_start
            all_scores[batch_start:batch_end] = action[:actual_count]

        return all_scores

    def _score_cross_section(
        self,
        stock_features: np.ndarray,
        portfolio_features: np.ndarray,
    ) -> np.ndarray:
        if self.use_custom_policy:
            return self._score_custom_policy(stock_features,
                                             portfolio_features)
        return self._score_mlp_policy(stock_features, portfolio_features)

    def predict_signals(
        self,
        df: pd.DataFrame,
        top_k: Optional[int] = None,
        return_details: bool = False,
        output_mode: Optional[str] = None,
        include_stock_scores: bool = False,
    ) -> pd.DataFrame:
        if top_k is not None and top_k < 0:
            raise ValueError(f"top_k_override must be >= 0, got {top_k}")

        mode = (output_mode or self.output_mode).lower()
        if mode not in {"trading", "rankic"}:
            raise ValueError(
                f"output_mode must be 'trading' or 'rankic', got {mode}")

        work = self._prepare_data(df)
        grouped = {t: g for t, g in work.groupby("trade_time", sort=True)}
        unique_times = sorted(grouped.keys())

        asset_ids = sorted(work["code"].unique().tolist())
        code_to_index = {c: i for i, c in enumerate(asset_ids)}

        prev_weights = np.zeros(len(asset_ids), dtype=np.float32)
        last_turnover = 0.0

        total_turnover = 0.0
        total_cost = 0.0
        total_portfolio_return = 0.0
        net_total_portfolio_return = 0.0

        results = []
        stock_predictions = []

        for step_idx, t in enumerate(unique_times):
            cs_data = grouped[t]
            codes = cs_data["code"].tolist()
            code_indices = np.array([code_to_index[c] for c in codes],
                                    dtype=np.int64)
            stock_features = cs_data[self.features].values.astype(np.float32)
            returns = cs_data["nxt1_ret"].values.astype(np.float32)

            portfolio_features = self._build_portfolio_features(
                prev_weights, last_turnover)
            scores = self._score_cross_section(stock_features,
                                               portfolio_features)

            if mode == "rankic":
                daily_ic = rank_ic(scores, returns) if len(scores) > 1 else 0.0
                row = {
                    "trade_time": t,
                    "rank_ic": float(daily_ic),
                    "n_assets": int(len(cs_data)),
                }
                if return_details:
                    row["score_mean"] = float(np.mean(scores))
                    row["score_std"] = float(np.std(scores))
                results.append(row)

                if include_stock_scores:
                    for i, code in enumerate(codes):
                        stock_predictions.append({
                            "trade_time": t,
                            "code": code,
                            "score": float(scores[i]),
                            "nxt1_ret": float(returns[i]),
                        })
                continue

            # trading 模式
            should_rebalance = False
            subset_weights = np.zeros(len(codes), dtype=np.float32)
            turnover = 0.0
            cost = 0.0
            portfolio_return = 0.0
            net_portfolio_return = 0.0

            if self.action_mode == "weights":
                should_rebalance = (
                    self.signal_config.rebalance_window <= 1
                    or (step_idx % self.signal_config.rebalance_window == 0))

                if should_rebalance:
                    weights_top_k = top_k if top_k is not None else 0
                    subset_weights = scores_to_weights(scores,
                                                       self.signal_config,
                                                       weights_top_k)
                    new_weights = np.zeros_like(prev_weights)
                    new_weights[code_indices] = subset_weights
                else:
                    new_weights = prev_weights.copy()
                    subset_weights = new_weights[code_indices]

                turnover = calculate_turnover(prev_weights, new_weights)
                cost = calculate_transaction_cost(prev_weights, new_weights,
                                                  self.signal_config)
                portfolio_return = calculate_portfolio_return(
                    subset_weights, returns)
                net_portfolio_return = portfolio_return - cost

                total_turnover += turnover
                total_cost += cost
                total_portfolio_return += portfolio_return
                net_total_portfolio_return += net_portfolio_return
                last_turnover = float(turnover)

                if self.include_portfolio_state:
                    drifted = subset_weights * (1.0 + returns)
                    if np.sum(drifted) > 0:
                        drifted = drifted / np.sum(drifted)
                    else:
                        drifted = np.zeros_like(drifted)
                    next_prev = np.zeros_like(prev_weights)
                    next_prev[code_indices] = drifted
                    prev_weights = next_prev
                else:
                    prev_weights = np.zeros_like(prev_weights)
            else:
                # raw_ic 直接输出IC，不走交易路径
                daily_ic = rank_ic(scores, returns) if len(scores) > 1 else 0.0
                prev_weights = np.zeros_like(prev_weights)
                last_turnover = 0.0

            row = {
                "trade_time":
                t,
                "portfolio_return":
                float(portfolio_return),
                "cost":
                float(cost),
                "turnover":
                float(turnover),
                "n_holdings":
                int(np.sum(subset_weights > 1e-8)),
                "hhi":
                float(np.sum(subset_weights**2)),
                "net_portfolio_return":
                float(net_portfolio_return),
                "net_total_portfolio_return":
                float(net_total_portfolio_return),
                "rank_ic":
                float(rank_ic(scores, returns) if len(scores) > 1 else 0.0),
                "rebalanced":
                bool(should_rebalance),
                "top_k_used":
                int(top_k) if top_k is not None else 0,
                "action_mode":
                self.action_mode,
            }
            if return_details:
                row["top_scores"] = str(np.sort(scores)[::-1][:5].tolist())
                row["total_turnover"] = float(total_turnover)
                row["total_cost"] = float(total_cost)
                row["total_portfolio_return"] = float(total_portfolio_return)
            results.append(row)

        self.stock_scores_df = pd.DataFrame(
            stock_predictions) if stock_predictions else None
        return pd.DataFrame(results)


def predict_test_set(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    top_k: Optional[int] = None,
    output_path: Optional[str] = None,
    deterministic: bool = True,
    return_details: bool = True,
    output_mode: Optional[str] = None,
    save_stock_scores: bool = True,
) -> pd.DataFrame:
    generator = TradingSignalGenerator(
        model_path=model_path,
        config_path=config_path,
        deterministic=deterministic,
    )
    pdb.set_trace()
    resolved_output_mode = (output_mode or generator.output_mode).lower()
    include_stock_scores = save_stock_scores and resolved_output_mode == "rankic"

    signals_df = generator.predict_signals(
        df=test_df,
        top_k=top_k,
        return_details=return_details,
        output_mode=resolved_output_mode,
        include_stock_scores=include_stock_scores,
    )
    pdb.set_trace()

    stock_df = generator.stock_scores_df
    stock_df = stock_df.set_index(['trade_time','code']).unstack()
    
    returns = stock_df['nxt1_ret']
    factors = stock_df['score']
    out_dir = os.path.dirname(output_path)
    pdb.set_trace()
    result = Metrics.quick(returns=returns,
                           factors=factors,
                           factor_name='rl',
                           hold=1,
                           skip=0,
                           category=1,
                           max_points=200,
                           save_file=os.path.join(out_dir, "plot.png"),
                           quantiles=10,
                           is_series=True)
    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        score_path = output_path.replace(
                ".csv", f"_{resolved_output_mode}.csv")
        signals_df.to_csv(score_path, index=False)
        print(f"预测结果已保存: {score_path}")

        if include_stock_scores and generator.stock_scores_df is not None:
            score_path = output_path.replace(
                ".csv", f"_{resolved_output_mode}_scores.csv")
            generator.stock_scores_df.to_csv(score_path, index=False)
            print(f"打分已保存: {score_path}")

    return signals_df
