import json, os, pdb
from typing import Optional

import numpy as np
import pandas as pd
import torch as th

from kichaos.stable3.sac import SAC

from lib.rl033.signal import (
    Config,
    calculate_portfolio_return,
    calculate_transaction_cost,
    calculate_turnover,
    rank_ic,
    scores_to_market_neutral_weights,
)


class TradingSignalGenerator:

    def __init__(
        self,
        model_path: str,
        config_path: str,
        deterministic: bool = True,
        env_config_override: Optional[dict] = None,
        signal_config_override: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.deterministic = deterministic

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.features = self.config["features"]
        self.env_config = self.config["env_config"]
        if env_config_override:
            self.env_config.update(env_config_override)

        self.use_custom_policy = self.config["use_custom_policy"]

        self.sampling_mode = self.env_config.get("sampling_mode",
                                                 "sequential").lower()
        self.action_mode = self.env_config.get("action_mode",
                                               "weights").lower()
        self.output_mode = self.env_config.get("output_mode",
                                               "trading").lower()
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

        sig = self.config.get("signal_config", {})
        if signal_config_override:
            sig.update(signal_config_override)

        self.signal_config = Config(
            min_weight=sig.get("min_weight", 0.0),
            max_weight=sig.get("max_weight", 0.05),
            normalize=sig.get("normalize", True),
            top_k=sig.get("top_k", 20),
            cost_rate=sig.get("cost_rate", 0.001),
            turnover_penalty=1.0,#sig.get("turnover_penalty", 0.0),
            rebalance_window=sig.get("rebalance_window", 1),
            softmax_temperature=sig.get("softmax_temperature", 0.1),
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

    def _score_custom_policy(self, stock_features: np.ndarray) -> np.ndarray:
        activation = "sigmoid"#"tanh" if self.action_mode == "raw_ic" else "sigmoid"
        with th.no_grad():
            stocks_t = th.as_tensor(stock_features,
                                    device=self.model.device,
                                    dtype=th.float32)

            scores = self.actor.score_assets(
                stock_features=stocks_t,
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
    
    def _score_cross_section(self, stock_features: np.ndarray) -> np.ndarray:
        if self.use_custom_policy:
            return self._score_custom_policy(stock_features=stock_features)
        portfolio_features = self._build_portfolio_features(
            prev_weights=np.zeros(stock_features.shape[0], dtype=np.float32),
            last_turnover=0.0,
        )
        return self._score_mlp_policy(
            stock_features=stock_features,
            portfolio_features=portfolio_features,
        )

    @staticmethod
    def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
        if returns is None or len(returns) < 2:
            return 0.0
        std = float(returns.std(ddof=0))
        if std <= 1e-12:
            return 0.0
        mean = float(returns.mean())
        return mean / std * np.sqrt(periods_per_year)


    def predict_signals(self, df:pd.DataFrame):
        work = self._prepare_data(df)
        grouped = {t: g for t, g in work.groupby("trade_time", sort=True)}
        unique_times = sorted(grouped.keys())
        asset_ids = sorted(work["code"].unique().tolist())
        code_to_index = {c: i for i, c in enumerate(asset_ids)}
        code_predictions = []
        for step_idx, t in enumerate(unique_times):
            cs_data = grouped[t]
            codes = cs_data["code"].tolist()
            code_indices = np.array([code_to_index[c] for c in codes],
                                    dtype=np.int64)
            stock_features = cs_data[self.features].values.astype(np.float32)
            returns = cs_data["nxt1_ret"].values.astype(np.float32)
            raw_scores = self._score_cross_section(stock_features)
            for i, (_, row) in enumerate(cs_data.iterrows()):
                code_predictions.append({
                    'trade_time': t,
                    'code': row['code'],
                    'score': float(raw_scores[i]),
                    'nxt1_ret': float(returns[i]),
                })
        code_pred_df = pd.DataFrame(code_predictions) if code_predictions else pd.DataFrame()
        
        return code_pred_df
        
    # def predict_signals1(
    #     self,
    #     df: pd.DataFrame,
    #     top_k: Optional[int] = None,
    #     return_details: bool = False,
    # ):
    #     if top_k is not None and top_k < 0:
    #         raise ValueError(f"top_k_override must be >= 0, got {top_k}")

    #     work = self._prepare_data(df)
    #     grouped = {t: g for t, g in work.groupby("trade_time", sort=True)}
    #     unique_times = sorted(grouped.keys())

    #     asset_ids = sorted(work["code"].unique().tolist())
    #     code_to_index = {c: i for i, c in enumerate(asset_ids)}

    #     prev_weights = np.zeros(len(asset_ids), dtype=np.float32)
    #     total_turnover = 0.0
    #     total_cost = 0.0
    #     total_ls_return = 0.0
    #     total_net_return = 0.0

    #     stock_predictions = []
    #     daily_results = []

    #     k = top_k if top_k is not None and top_k > 0 else self.signal_config.top_k

    #     for step_idx, t in enumerate(unique_times):
    #         print(t)
    #         cs_data = grouped[t]
    #         codes = cs_data["code"].tolist()
    #         code_indices = np.array([code_to_index[c] for c in codes],
    #                                 dtype=np.int64)
    #         stock_features = cs_data[self.features].values.astype(np.float32)
    #         returns = cs_data["nxt1_ret"].values.astype(np.float32)

    #         raw_scores = self._score_cross_section(stock_features)
    #         daily_ic = rank_ic(raw_scores, returns)

    #         # 1) raw_scores -> Top多/Bottom空权重 -> 多空收益
    #         neutral_weights_cs = scores_to_market_neutral_weights(raw_scores, top_k=k)
    #         ls_return = calculate_portfolio_return(neutral_weights_cs, returns)

    #         # 2) raw_scores -> 中性权重 -> 扣费后收益 -> 净值/Sharpe
    #         new_weights_full = np.zeros_like(prev_weights)
    #         new_weights_full[code_indices] = neutral_weights_cs
    #         turnover = calculate_turnover(prev_weights, new_weights_full)
            
    #         cost = calculate_transaction_cost(prev_weights, new_weights_full, self.signal_config)
    #         net_return = ls_return - cost

    #         total_turnover += turnover
    #         total_cost += cost
    #         total_ls_return += ls_return
    #         total_net_return += net_return

    #         prev_weights = new_weights_full

    #         long_mask = neutral_weights_cs > 0
    #         short_mask = neutral_weights_cs < 0
    #         long_leg_ret = float(np.mean(returns[long_mask])) if np.any(long_mask) else 0.0
    #         short_leg_ret = float(np.mean(returns[short_mask])) if np.any(short_mask) else 0.0

    #         for i, (_, row) in enumerate(cs_data.iterrows()):
    #             stock_predictions.append({
    #                 'trade_time': t,
    #                 'code': row['code'],
    #                 'score': float(raw_scores[i]),
    #                 'nxt1_ret': float(returns[i]),
    #             })

    #         daily_results.append({
    #             'trade_time': t,
    #             'rank_ic': float(daily_ic),
    #             'n_stocks': int(len(cs_data)),
    #             'top_k_used': int(k),
    #             'long_leg_ret': float(long_leg_ret),
    #             'short_leg_ret': float(short_leg_ret),
    #             'long_short_ret': float(ls_return),
    #             'turnover': float(turnover),
    #             'cost': float(cost),
    #             'net_ret': float(net_return),
    #         })

    #     stock_pred_df = pd.DataFrame(stock_predictions) if stock_predictions else pd.DataFrame()
    #     daily_rank_ic = pd.DataFrame(daily_results) if daily_results else pd.DataFrame()

    #     if len(daily_rank_ic) > 0:
    #         daily_rank_ic['gross_nav'] = (1.0 + daily_rank_ic['long_short_ret']).cumprod()
    #         daily_rank_ic['net_nav'] = (1.0 + daily_rank_ic['net_ret']).cumprod()

    #         mean_ic = daily_rank_ic['rank_ic'].mean()
    #         std_ic = daily_rank_ic['rank_ic'].std()
    #         ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0  # IC_IR：IC 均值/IC 标准差，衡量 IC 稳定性
    #         sharpe_gross = self._annualized_sharpe(daily_rank_ic['long_short_ret'])
    #         sharpe_net = self._annualized_sharpe(daily_rank_ic['net_ret'])
    #         avg_turnover = float(daily_rank_ic['turnover'].mean())
    #         avg_cost = float(daily_rank_ic['cost'].mean())

    #         print(f"平均每日 RankIC: {mean_ic:.6f} ± {std_ic:.6f}")
    #         print(f"IC_IR (稳定性): {ic_ir:.4f}")
    #         print(f"最大 RankIC: {daily_rank_ic['rank_ic'].max():.6f}")
    #         print(f"最小 RankIC: {daily_rank_ic['rank_ic'].min():.6f}")
    #         print(f"RankIC > 0 的天数占比：{(daily_rank_ic['rank_ic'] > 0).sum() / len(daily_rank_ic):.2%}")
    #         print(f"多空收益 Sharpe(扣费前): {sharpe_gross:.4f}")
    #         print(f"多空收益 Sharpe(扣费后): {sharpe_net:.4f}")
    #         print(f"平均换手: {avg_turnover:.6f}, 平均成本: {avg_cost:.6f}")
    #         print(f"累计多空收益(扣费前): {total_ls_return:.6f}")
    #         print(f"累计多空收益(扣费后): {total_net_return:.6f}")

    #     return stock_pred_df, daily_rank_ic

def predict_test_set(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    output_path: Optional[str] = None,
    deterministic: bool = True,
    env_config_override: Optional[dict] = None,
    signal_config_override: Optional[dict] = None,
) -> pd.DataFrame:
    generator = TradingSignalGenerator(
        model_path=model_path,
        config_path=config_path,
        deterministic=deterministic,
        env_config_override=env_config_override,
        signal_config_override=signal_config_override)
    
    code_pred_df = generator.predict_signals(df=test_df)
    pdb.set_trace()
    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        os.makedirs(out_dir, exist_ok=True)
    print(output_path)
    code_pred_df.to_csv(output_path, index=False)
    

    # stock_pred_df, daily_rank_ic = generator.predict_signals(df=test_df,
    #                                        top_k=top_k,
    #                                        return_details=return_details)

    # if output_path is not None:
    #     out_dir = os.path.dirname(output_path)
    #     if out_dir:
    #         os.makedirs(out_dir, exist_ok=True)
    #     daily_rank_ic.to_csv(output_path, index=False)
    #     print(f"预测结果已保存: {output_path}")
    #     pdb.set_trace()
    #     if stock_pred_df is not None:
    #         score_path = output_path.replace(".csv", "_stock_scores.csv")
    #         stock_pred_df.to_csv(score_path, index=False)
    #         print(f"股票打分已保存: {score_path}")
