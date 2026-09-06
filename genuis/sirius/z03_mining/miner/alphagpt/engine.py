"""使用 REINFORCE 训练、以 Polars 执行公式和评分的挖掘引擎。"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import polars as pl

from miner.evaluate import (
    EvaluationType,
    TimeSeriesEvaluationConfig,
    detect_evaluation_type,
    ensure_evaluation_supported,
    evaluate_formula,
)

from .config import AlphaGPTConfig
from .ops import OperatorSpec
from .vm import StackVM


class AlphaEngine:
    def __init__(
        self,
        *,
        model: Any,
        df_lazy: pl.LazyFrame,
        feature_names: tuple[str, ...],
        return_column: str,
        operator_specs: tuple[OperatorSpec, ...],
        config: AlphaGPTConfig,
        device: Any,
        evaluation_type: EvaluationType | None = None,
        evaluation_config: TimeSeriesEvaluationConfig | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.df_lazy = df_lazy
        self.return_column = return_column
        self.config = config
        self.device = device
        self.evaluation_type = evaluation_type or detect_evaluation_type(df_lazy)
        ensure_evaluation_supported(self.evaluation_type)
        self.evaluation_config = evaluation_config or TimeSeriesEvaluationConfig()
        self.vm = StackVM(feature_names, operator_specs)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate
        )
        self.metric_cache: dict[str, dict[str, float | int]] = {}
        self.discoveries: dict[str, dict[str, Any]] = {}
        self.best_score = -math.inf
        self.evaluation_count = 0
        self.cache_hit_count = 0
        self.invalid_formula_count = 0
        self.evaluation_error_count = 0

    def compute_reward(
        self, formula_tokens: list[int]
    ) -> tuple[float, Any, dict[str, float | int] | None]:
        decoded = self.vm.decode(formula_tokens)
        if decoded is None:
            self.invalid_formula_count += 1
            return self.config.invalid_reward, None, None
        formula_id = decoded.formula.factor_id
        metrics = self.metric_cache.get(formula_id)
        if metrics is None:
            self.evaluation_count += 1
            try:
                metrics = evaluate_formula(
                    self.df_lazy,
                    decoded.formula,
                    return_column=self.return_column,
                    min_observations=self.config.min_observations,
                    compiler=self.vm.compiler,
                    evaluation_type=self.evaluation_type,
                    time_series_config=self.evaluation_config,
                )
            except (
                pl.exceptions.PolarsError,
                ValueError,
                ZeroDivisionError,
                OverflowError,
            ):
                self.evaluation_error_count += 1
                metrics = {
                    "ic_mean": 0.0,
                    "abs_ic_mean": 0.0,
                    "ic_sharpe": 0.0,
                    "observations": 0,
                    "period_count": 0,
                }
            self.metric_cache[formula_id] = metrics
        else:
            self.cache_hit_count += 1
        return float(metrics[self.config.score]), decoded, metrics

    def train(self) -> list[dict[str, Any]]:
        torch = self.torch
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.model.train()
        started_at = time.perf_counter()

        for step in range(self.config.train_steps):
            completed_steps = step + 1
            log_this_step = (
                self.config.verbose
                and (
                    completed_steps == 1
                    or completed_steps % self.config.log_interval == 0
                    or completed_steps == self.config.train_steps
                )
            )
            if log_this_step:
                print(
                    f"[AlphaGPT] 开始第{completed_steps}/"
                    f"{self.config.train_steps}轮采样与评估",
                    flush=True,
                )
            tokens = torch.full(
                (self.config.batch_size, 1),
                self.model.bos_token,
                dtype=torch.long,
                device=self.device,
            )
            stack_depths = [0] * self.config.batch_size
            sampled_tokens: list[Any] = []
            log_probs: list[Any] = []

            for position in range(self.config.max_formula_len):
                logits, _ = self.model(tokens)
                masks = []
                for row, depth in enumerate(stack_depths):
                    valid_mask = self.vm.valid_token_mask(
                        depth,
                        self.config.max_formula_len - position - 1,
                    )
                    masks.append(valid_mask)
                mask = torch.tensor(
                        masks,
                        dtype=torch.bool,
                        device=self.device,
                    )
                logits = logits.masked_fill(~mask, -torch.inf)
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                sampled_tokens.append(action)
                log_probs.append(distribution.log_prob(action))
                for row, token in enumerate(action.tolist()):
                    stack_depths[row] = self.vm.next_stack_depth(
                        stack_depths[row], token
                    )
                tokens = torch.cat((tokens, action.unsqueeze(1)), dim=1)

            sequences = torch.stack(sampled_tokens, dim=1)
            rewards = torch.zeros(self.config.batch_size, device=self.device)
            batch_log_interval = max(1, self.config.batch_size // 4)
            for row in range(self.config.batch_size):
                token_ids = sequences[row].tolist()
                score, decoded, metrics = self.compute_reward(token_ids)
                rewards[row] = score
                evaluated_rows = row + 1
                if log_this_step and (
                    evaluated_rows % batch_log_interval == 0
                    or evaluated_rows == self.config.batch_size
                ):
                    print(
                        f"[AlphaGPT] 第{completed_steps}轮候选评估="
                        f"{evaluated_rows}/{self.config.batch_size}",
                        flush=True,
                    )
                if decoded is None or metrics is None:
                    continue
                formula_id = decoded.formula.factor_id
                record = {
                    "step": step,
                    "score": score,
                    "formula_id": formula_id,
                    "formula": decoded.expression,
                    "features": sorted(decoded.formula.features),
                    "tokens": list(decoded.tokens),
                    **metrics,
                }
                previous = self.discoveries.get(formula_id)
                if previous is None or score > previous["score"]:
                    self.discoveries[formula_id] = record
                self.best_score = max(self.best_score, score)

            reward_std = rewards.std(unbiased=False)
            advantages = (rewards - rewards.mean()) / (reward_std + 1e-5)
            loss = sum(
                (-log_probability * advantages).mean()
                for log_probability in log_probs
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            should_log = (
                self.config.verbose
                and (
                    completed_steps == 1
                    or completed_steps % self.config.log_interval == 0
                    or completed_steps == self.config.train_steps
                )
            )
            if should_log:
                elapsed = time.perf_counter() - started_at
                steps_per_second = completed_steps / elapsed if elapsed > 0 else 0.0
                remaining = self.config.train_steps - completed_steps
                eta = remaining / steps_per_second if steps_per_second > 0 else 0.0
                print(
                    f"[AlphaGPT] 进度={completed_steps}/{self.config.train_steps} "
                    f"平均奖励={rewards.mean().item():.6f} "
                    f"最佳奖励={max(self.best_score, 0.0):.6f} "
                    f"有效候选={len(self.discoveries)} "
                    f"实际评估={self.evaluation_count} "
                    f"缓存命中={self.cache_hit_count} "
                    f"无效公式={self.invalid_formula_count} "
                    f"评估异常={self.evaluation_error_count} "
                    f"耗时={elapsed:.1f}秒 预计剩余={eta:.1f}秒",
                    flush=True,
                )

        return sorted(
            self.discoveries.values(),
            key=lambda item: item["score"],
            reverse=True,
        )


__all__ = ["AlphaEngine"]
