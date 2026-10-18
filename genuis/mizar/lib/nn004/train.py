"""多资产 HybridTransformer 可配置损失训练与完整校验集 IC 选模。"""

import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch as th
from torch import nn

from .data import SequenceStore, balanced_batches
from .inference import load_checkpoint, predict_store, resolve_device
from .losses import multi_asset_loss, validate_loss_config
from .metrics import evaluate_predictions
from .model import HybridTransformerRegressor


VERSION = "hybrid_transformer_loss_v1"


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, th.device):
        return str(value)
    raise TypeError(type(value).__name__)


def _summary(result):
    return {key: value for key, value in result.items()
            if key != "ic_sequence"}


def _set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    th.manual_seed(int(seed))
    if th.cuda.is_available():
        th.cuda.manual_seed_all(int(seed))


def _sanitize_frame(frame, features):
    out = frame.copy()
    missing = {"trade_time", "code", "nxt1_ret", *features} - set(out.columns)
    if missing:
        raise ValueError(f"数据缺少字段: {sorted(missing)}")
    out[features] = out[features].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(out[features].to_numpy(dtype=np.float64))
    if int(bad.sum()):
        print(f"[WARN] 特征中发现 {int(bad.sum())} 个 NaN/Inf，已填充为 0.0")
    out[features] = out[features].replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _architecture(n_features, lookback, prediction_bound_std, config):
    return {
        "n_features": int(n_features),
        "lookback": int(lookback),
        "d_model": int(config.get("d_model", 64)),
        "n_heads": int(config.get("n_heads", 4)),
        "e_layers": int(config.get("e_layers", 2)),
        "d_layers": int(config.get("d_layers", 1)),
        "d_ff": int(config.get("d_ff", 128)),
        "dropout": float(config.get("dropout", 0.1)),
        "activation": str(config.get("activation", "gelu")),
        "prediction_bound_std": float(prediction_bound_std),
    }


def _read_loss_config(train_config):
    required = {"loss_type", "huber_delta", "corr_weight"}
    missing = required - set(train_config)
    if missing:
        raise ValueError(
            f"HybridTransformer-Loss 参数缺少: {sorted(missing)}；"
            "禁止静默回退到旧损失")
    return validate_loss_config(
        train_config["loss_type"], train_config["huber_delta"],
        train_config["corr_weight"], train_config.get("corr_eps", 1e-8))


def _objective_name(loss_config):
    if loss_config["loss_type"] == "mse":
        return "standardized_return_multi_asset_mse"
    return "standardized_return_multi_asset_huber_corr"


def _save_checkpoint(path, model, optimizer, architecture, features,
                     lookback, scales, epoch, score, train_metrics,
                     loss_config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    th.save({
        "version": VERSION,
        "objective": _objective_name(loss_config),
        "loss_config": dict(loss_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": architecture,
        "features": list(features),
        "lookback": int(lookback),
        "ret_scale_by_code": dict(scales),
        "epoch": int(epoch),
        "selection_score": float(score),
        "train_metrics": dict(train_metrics),
    }, path)


def _evaluate(model, store, batch_size, device, resampling_minutes,
              roll_window, min_periods, label):
    started = time.monotonic()
    print(f"[{label}_START] rows={len(store.df)} batch_size={batch_size} "
          f"device={device}", flush=True)
    records = predict_store(model, store, batch_size, device)
    inference_seconds = time.monotonic() - started
    result = evaluate_predictions(
        records, resampling_minutes=resampling_minutes,
        roll_window=roll_window, min_periods=min_periods)
    result["rows"] = int(len(records))
    result["inference_seconds"] = float(inference_seconds)
    result["elapsed_seconds"] = float(time.monotonic() - started)
    print(f"[{label}_END] score={result['selection_score']:.6f} "
          f"elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    return result


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                features: List[str], env_config: Dict[str, Any],
                model_config: Dict[str, Any], train_config: Dict[str, Any],
                output_dir: str) -> Tuple[HybridTransformerRegressor,
                                          Dict[str, Any]]:
    if not output_dir:
        raise ValueError("output_dir 不能为空")
    features = list(features)
    lookback = int(env_config.get("default_lookback", 20))
    bound = float(env_config.get("prediction_bound_std", 3.0))
    seed = int(env_config.get("seed", 42))
    epochs = int(train_config.get("epochs", 20))
    batch_size = int(model_config.get("batch_size", 1024))
    inference_batch_size = int(train_config.get("inference_batch_size", 4096))
    samples_per_epoch = int(train_config.get("samples_per_epoch", 0)) or None
    eval_every = int(train_config.get("eval_every_epochs", 5))
    min_valid_ic = int(train_config.get("min_valid_rolling_ic", 20))
    min_evals = int(train_config.get("early_stop_min_evals", 3))
    patience = int(train_config.get("early_stop_patience_evals", 2))
    min_delta = float(train_config.get("early_stop_ic_min_delta", 0.0005))
    resampling = int(train_config.get("ic_resampling_minutes", 5))
    roll_window = int(train_config.get("ic_roll_window", 15))
    min_periods = int(train_config.get("ic_min_periods", 5))
    progress_seconds = float(train_config.get("progress_seconds", 15.0))
    loss_config = _read_loss_config(train_config)
    device = resolve_device(model_config.get("device", "auto"))
    use_amp = bool(model_config.get("use_amp", True)) and device.type == "cuda"
    if min(epochs, batch_size, eval_every, inference_batch_size) <= 0:
        raise ValueError("epochs、batch_size 和验证参数必须大于0")

    _set_seed(seed)
    if device.type == "cuda":
        th.backends.cudnn.deterministic = False
        th.backends.cudnn.benchmark = True
        th.backends.cuda.matmul.allow_tf32 = bool(
            model_config.get("allow_tf32", True))
        if hasattr(th, "set_float32_matmul_precision"):
            th.set_float32_matmul_precision("high")
    train_store = SequenceStore(
        _sanitize_frame(train_df, features), features, lookback,
        fit_return_scale=True)
    val_store = SequenceStore(
        _sanitize_frame(val_df, features), features, lookback,
        ret_scale_by_code=train_store.ret_scale_by_code)
    if set(train_store.asset_codes) != set(val_store.asset_codes):
        raise ValueError("训练集和校验集品种不一致")

    architecture = _architecture(len(features), lookback, bound, model_config)
    model = HybridTransformerRegressor(**architecture).to(device)
    optimizer = th.optim.AdamW(
        model.parameters(), lr=float(model_config.get("learning_rate", 5e-5)),
        weight_decay=float(model_config.get("weight_decay", 1e-5)))
    scaler = th.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip = float(model_config.get("grad_clip_norm", 1.0))
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    best_path = os.path.join(model_dir, "best_model.pt")
    final_path = os.path.join(model_dir, "final_model.pt")
    history_path = os.path.join(log_dir, "validation_history.jsonl")
    open(history_path, "w", encoding="utf-8").close()
    run_config = {
        "version": VERSION,
        "backbone": "kichaos.nn.HybridTransformer.HybridTransformer",
        "objective": _objective_name(loss_config),
        "loss_config": dict(loss_config),
        "selection_metric": "min_asset_rolling_pearson_ic_mean",
        "features": features,
        "lookback": lookback,
        "ret_scale_by_code": train_store.ret_scale_by_code,
        "asset_id_by_code": train_store.asset_id_by_code,
        "env_config": dict(env_config),
        "model_config": dict(model_config),
        "architecture": architecture,
        "train_config": dict(train_config),
        "train_size": int(len(train_store.df)),
        "val_size": int(len(val_store.df)),
        "device": str(device),
        "training_date": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "config.json"), "w",
              encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2,
                  default=_json_default)

    print(f"[HYBRID_TRANSFORMER_LOSS_INIT] device={device} amp={use_amp} "
          f"lookback={lookback} features={len(features)} batch={batch_size} "
          f"epochs={epochs} loss={loss_config} architecture={architecture}",
          flush=True)
    print(f"[HYBRID_TRANSFORMER_LOSS_DATA] train={len(train_store.df)} "
          f"val={len(val_store.df)} assets={train_store.asset_codes} "
          f"asset_ids={train_store.asset_id_by_code} "
          f"train_std={train_store.ret_scale_by_code}", flush=True)
    best_score, best_epoch = -np.inf, 0
    eval_count = no_improve_count = 0
    early_reference = -np.inf
    stopped_early = False

    for epoch in range(1, epochs + 1):
        model.train()
        started = progress_at = time.monotonic()
        # 留在GPU上累计，避免每个batch为日志执行四次GPU->CPU同步。
        metric_sums = th.zeros(4, dtype=th.float64, device=device)
        sample_count = batch_count = 0
        for positions in balanced_batches(
                train_store, batch_size, epoch, seed, samples_per_epoch):
            x = th.from_numpy(train_store.observation_batch(positions)).to(
                device, non_blocking=True)
            y = th.from_numpy(train_store.target_z_batch(positions)).to(
                device, non_blocking=True)
            asset_ids = th.from_numpy(
                train_store.asset_id_batch(positions)).to(
                    device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with th.cuda.amp.autocast(enabled=use_amp):
                prediction = model(x)
            # 相关性包含中心化、方差和除法，统一在FP32中计算，避免AMP不稳定。
            loss, components = multi_asset_loss(
                prediction.float(), y.float(), asset_ids,
                **loss_config)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            count = len(positions)
            metric_sums += th.stack([
                loss.detach(), components["point_loss"].detach(),
                components["corr_loss"].detach(),
                components["mean_batch_corr"].detach(),
            ]).to(dtype=th.float64) * count
            sample_count += count
            batch_count += 1
            now = time.monotonic()
            if now - progress_at >= progress_seconds:
                denom = max(sample_count, 1)
                values = (metric_sums / denom).detach().cpu().tolist()
                if not np.isfinite(values).all():
                    raise FloatingPointError(
                        f"训练损失出现 NaN/Inf: {values}")
                print(f"[HYBRID_TRANSFORMER_LOSS_PROGRESS] "
                      f"epoch={epoch}/{epochs} batches={batch_count} "
                      f"samples={sample_count} loss={values[0]:.6f} "
                      f"point={values[1]:.6f} corr={values[3]:.6f} "
                      f"speed={sample_count/(now-started):.0f}/s", flush=True)
                progress_at = now
        denom = max(sample_count, 1)
        values = (metric_sums / denom).detach().cpu().tolist()
        if not np.isfinite(values).all():
            raise FloatingPointError(f"训练损失出现 NaN/Inf: {values}")
        train_metrics = {
            "train_loss": values[0],
            "train_point_loss": values[1],
            "train_corr_loss": values[2],
            "train_batch_corr": values[3],
        }
        print(f"[HYBRID_TRANSFORMER_LOSS_EPOCH] epoch={epoch}/{epochs} "
              f"loss={train_metrics['train_loss']:.6f} "
              f"point={train_metrics['train_point_loss']:.6f} "
              f"corr={train_metrics['train_batch_corr']:.6f} "
              f"samples={sample_count} elapsed={time.monotonic()-started:.1f}s",
              flush=True)
        if epoch % eval_every and epoch != epochs:
            continue
        eval_count += 1
        result = _evaluate(model, val_store, inference_batch_size, device,
                           resampling, roll_window, min_periods,
                           f"HYBRID_TRANSFORMER_LOSS_VAL_E{epoch}")
        score = float(result["selection_score"])
        eligible = np.isfinite(score) and bool(result["assets"])
        for metrics in result["assets"].values():
            eligible = (eligible and
                        metrics["valid_rolling_ic"] >= min_valid_ic and
                        metrics["prediction_std"] > 1e-12)
        improved = bool(eligible and score > best_score)
        if improved:
            best_score, best_epoch = score, epoch
            _save_checkpoint(
                best_path, model, optimizer, architecture, features, lookback,
                train_store.ret_scale_by_code, epoch, score, train_metrics,
                loss_config)
            print(f"[HYBRID_TRANSFORMER_LOSS_BEST] epoch={epoch} "
                  f"score={score:.6f} path={best_path}", flush=True)
        record = {"epoch": epoch, "eval_count": eval_count,
                  **train_metrics, "eligible": bool(eligible),
                  "saved_as_best": improved, **_summary(result)}
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False,
                                    default=_json_default) + "\n")
        if eligible:
            if score > early_reference + min_delta:
                early_reference, no_improve_count = score, 0
            else:
                no_improve_count += 1
            if eval_count >= min_evals and no_improve_count >= patience:
                stopped_early = True
                print(f"[HYBRID_TRANSFORMER_LOSS_EARLY_STOP] epoch={epoch} "
                      f"best_epoch={best_epoch} best_score={best_score:.6f}",
                      flush=True)
                break

    _save_checkpoint(
        final_path, model, optimizer, architecture, features, lookback,
        train_store.ret_scale_by_code, epoch,
        best_score if np.isfinite(best_score) else np.nan,
        train_metrics, loss_config)
    if not os.path.isfile(best_path):
        raise RuntimeError("没有产生符合要求的最佳模型")
    checkpoint = load_checkpoint(best_path, device)
    best_model = HybridTransformerRegressor(
        **checkpoint["model_config"]).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    final_result = _evaluate(
        best_model, val_store, inference_batch_size, device,
        resampling, roll_window, min_periods,
        "HYBRID_TRANSFORMER_LOSS_FINAL")
    final_summary = {"model": "best_ic", "best_epoch": best_epoch,
                     "loss_config": dict(loss_config),
                     **_summary(final_result)}
    final_validation_path = os.path.join(
        log_dir, "final_full_ic_validation.json")
    with open(final_validation_path, "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, ensure_ascii=False, indent=2,
                  default=_json_default)
    final_result["ic_sequence"].to_csv(
        os.path.join(log_dir, "final_rolling_ic.csv"), index=False)
    info = {
        "best_model_path": best_path,
        "final_model_path": final_path,
        "config_path": os.path.join(output_dir, "config.json"),
        "validation_history_path": history_path,
        "final_validation_path": final_validation_path,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "loss_config": dict(loss_config),
        "stopped_early": stopped_early,
    }
    print(f"[HYBRID_TRANSFORMER_LOSS_END] best_epoch={best_epoch} "
          f"best_score={best_score:.6f} stopped_early={stopped_early}",
          flush=True)
    return best_model, info


__all__ = ["train_model"]
