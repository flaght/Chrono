"""多资产标准化收益损失：逐品种计算，再等权聚合。"""

from typing import Any, Dict, Tuple

import torch as th


SUPPORTED_LOSS_TYPES = {"mse", "huber_corr"}


def validate_loss_config(loss_type: str, huber_delta: float,
                         corr_weight: float, corr_eps: float) -> Dict[str, Any]:
    """校验并返回可写入配置、检查点的规范化损失参数。"""
    loss_type = str(loss_type).strip().lower()
    if loss_type not in SUPPORTED_LOSS_TYPES:
        raise ValueError(
            f"loss_type 必须是 {sorted(SUPPORTED_LOSS_TYPES)}，实际为 {loss_type!r}")
    huber_delta = float(huber_delta)
    corr_weight = float(corr_weight)
    corr_eps = float(corr_eps)
    if huber_delta <= 0:
        raise ValueError("huber_delta 必须大于0")
    if corr_weight < 0:
        raise ValueError("corr_weight 不能小于0")
    if corr_eps <= 0:
        raise ValueError("corr_eps 必须大于0")
    if loss_type == "mse" and corr_weight != 0:
        raise ValueError("loss_type=mse 时 corr_weight 必须为0")
    return {
        "loss_type": loss_type,
        "huber_delta": huber_delta,
        "corr_weight": corr_weight,
        "corr_eps": corr_eps,
    }


def pearson_corr_loss(prediction: th.Tensor, target: th.Tensor,
                      eps: float = 1e-8) -> Tuple[th.Tensor, th.Tensor]:
    """返回 ``1-PearsonCorr`` 与相关系数；运算方应传入 FP32 张量。"""
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    if prediction.numel() < 2 or target.numel() != prediction.numel():
        raise ValueError("相关性损失要求 prediction/target 等长且至少包含2条")
    pred_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    numerator = th.sum(pred_centered * target_centered)
    denominator = th.sqrt(th.sum(pred_centered.square()) + float(eps)) * th.sqrt(
        th.sum(target_centered.square()) + float(eps))
    corr = th.clamp(numerator / denominator, min=-1.0, max=1.0)
    return 1.0 - corr, corr


def multi_asset_loss(prediction: th.Tensor, target: th.Tensor,
                     asset_ids: th.Tensor, loss_type: str = "mse",
                     huber_delta: float = 1.0, corr_weight: float = 0.0,
                     corr_eps: float = 1e-8):
    """
    对每个品种分别计算点损失和相关性损失，再对品种等权平均。

    这样即使未来批次中两个品种的条数略有差异，也不会让样本更多的品种
    支配目标。返回的组件均为标量张量，便于训练日志留痕。
    """
    config = validate_loss_config(
        loss_type, huber_delta, corr_weight, corr_eps)
    prediction = prediction.reshape(-1).float()
    target = target.reshape(-1).float()
    asset_ids = asset_ids.reshape(-1).to(device=prediction.device)
    if prediction.numel() != target.numel() or target.numel() != asset_ids.numel():
        raise ValueError("prediction、target、asset_ids 长度必须一致")
    if not prediction.numel():
        raise ValueError("损失不能在空批次上计算")
    point_parts = []
    corr_loss_parts = []
    corr_parts = []
    unique_assets = th.unique(asset_ids, sorted=True)
    if unique_assets.numel() < 2:
        raise ValueError("多资产损失要求每个批次至少包含两个品种")
    for asset_id in unique_assets:
        mask = asset_ids.eq(asset_id)
        pred_asset = prediction[mask]
        target_asset = target[mask]
        if pred_asset.numel() < 2:
            raise ValueError("每个品种在批次中至少需要2条样本")
        if config["loss_type"] == "mse":
            point = (pred_asset - target_asset).square().mean()
        else:
            # 手工写出标准Huber，避免依赖不同PyTorch版本的函数签名。
            absolute_error = (pred_asset - target_asset).abs()
            delta = config["huber_delta"]
            point = th.where(
                absolute_error <= delta,
                0.5 * absolute_error.square(),
                delta * (absolute_error - 0.5 * delta)).mean()
        corr_loss, corr = pearson_corr_loss(
            pred_asset, target_asset, eps=config["corr_eps"])
        point_parts.append(point)
        corr_loss_parts.append(corr_loss)
        corr_parts.append(corr)

    point_loss = th.stack(point_parts).mean()
    corr_loss = th.stack(corr_loss_parts).mean()
    mean_corr = th.stack(corr_parts).mean()
    total_loss = point_loss + config["corr_weight"] * corr_loss
    return total_loss, {
        "point_loss": point_loss,
        "corr_loss": corr_loss,
        "mean_batch_corr": mean_corr,
    }


__all__ = ["SUPPORTED_LOSS_TYPES", "multi_asset_loss", "pearson_corr_loss",
           "validate_loss_config"]
