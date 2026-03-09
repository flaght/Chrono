# -*- coding: utf-8 -*-
"""
算子定义 — 与 test/code/cython 逻辑对齐的 PyTorch 张量实现。

OPS_CONFIG 动态构建, 支持外部配置的 DEFAULT_PERIODS。
所有算子均基于 torch.Tensor，输入输出形状为 [N_tokens, T_steps]。

[优化版本]：使用 cumsum 和 conv1d 替代耗时的 for 循环和 unfold，极大提升计算速度。
"""
import torch
import torch.nn.functional as F
from .config import ModelConfig


# ── 工具函数 ─────────────
@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0:
        return x
    pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-d]], dim=1)


@torch.jit.script
def _ts_sum(x: torch.Tensor, w: int) -> torch.Tensor:
    """O(1) 复杂度的滑动窗口求和"""
    if w <= 1:
        return x
    cs = torch.cumsum(x, dim=1)
    # 在左侧填充 w 个 0
    pad = torch.zeros((x.shape[0], w), dtype=x.dtype, device=x.device)
    cs_pad = torch.cat([pad, cs], dim=1)
    # 窗口和 = cs[t] - cs[t-w]
    return cs_pad[:, w:] - cs_pad[:, :-w]


# ── 时间序列算子 (需 Window 参数) ─────────────
@torch.jit.script
def _ts_ema(x: torch.Tensor, w: int) -> torch.Tensor:
    """使用 1D 卷积极其高效地近似 EMA (截断至 5*w)"""
    alpha = 2.0 / (w + 1.0)
    k_len = int(min(x.shape[1], 5 * w))
    if k_len == 0:
        return x
        
    idx = torch.arange(k_len, device=x.device, dtype=x.dtype)
    kernel = alpha * torch.pow(1.0 - alpha, idx)
    # 翻转构建卷积核 [out_channels=1, in_channels=1, length]
    kernel = kernel.flip(0).view(1, 1, -1)
    
    # 输入形状 [N, 1, T] 填充左侧 k_len - 1 个 0
    x_pad = F.pad(x.unsqueeze(1), (k_len - 1, 0))
    res = F.conv1d(x_pad, kernel)
    return res.squeeze(1)


@torch.jit.script
def _ts_mean(x: torch.Tensor, w: int) -> torch.Tensor:
    if w <= 1:
        return x
    return _ts_sum(x, w) / float(w)


@torch.jit.script
def _ts_std(x: torch.Tensor, w: int) -> torch.Tensor:
    """O(1) 复杂度的滑动标准差"""
    if w <= 1:
        return torch.zeros_like(x)
    sum_x = _ts_sum(x, w)
    sum_x2 = _ts_sum(x * x, w)
    var = (sum_x2 - (sum_x * sum_x) / float(w)) / float(w - 1)
    return torch.clamp(var, min=0.0).sqrt()


@torch.jit.script
def _ts_skew(x: torch.Tensor, w: int) -> torch.Tensor:
    """O(1) 复杂度的滑动偏度"""
    if w <= 1:
        return torch.zeros_like(x)
    sum_x = _ts_sum(x, w)
    sum_x2 = _ts_sum(x * x, w)
    sum_x3 = _ts_sum(x * x * x, w)
    
    mu = sum_x / float(w)
    var = (sum_x2 - sum_x * mu) / float(w - 1)
    sigma = torch.clamp(var, min=1e-6).sqrt()
    
    m3 = (sum_x3 - 3.0 * mu * sum_x2 + 2.0 * float(w) * torch.pow(mu, 3.0)) / float(w)
    return m3 / (torch.pow(sigma, 3.0) + 1e-6)


@torch.jit.script
def _ts_kurt(x: torch.Tensor, w: int) -> torch.Tensor:
    """O(1) 复杂度的滑动峰度"""
    if w <= 1:
        return torch.zeros_like(x)
    sum_x = _ts_sum(x, w)
    sum_x2 = _ts_sum(x * x, w)
    sum_x3 = _ts_sum(x * x * x, w)
    sum_x4 = _ts_sum(x * x * x * x, w)
    
    mu = sum_x / float(w)
    var = (sum_x2 - sum_x * mu) / float(w - 1)
    sigma2 = torch.clamp(var, min=1e-6)
    
    m4 = (sum_x4 - 4.0 * mu * sum_x3 + 6.0 * torch.pow(mu, 2.0) * sum_x2 - 3.0 * float(w) * torch.pow(mu, 4.0)) / float(w)
    return m4 / (torch.pow(sigma2, 2.0) + 1e-6) - 3.0


@torch.jit.script
def _ts_rsi(x: torch.Tensor, w: int) -> torch.Tensor:
    delta = x - _ts_delay(x, 1)
    up = torch.clamp(delta, min=0.0)
    down = torch.clamp(-delta, min=0.0)
    
    roll_up = _ts_mean(up, w)
    roll_down = _ts_mean(down, w)
    
    rs = roll_up / (roll_down + 1e-6)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


@torch.jit.script
def _ts_corr(x: torch.Tensor, y: torch.Tensor, w: int) -> torch.Tensor:
    """O(1) 复杂度的滑动相关系数"""
    if w <= 1:
        return torch.zeros_like(x)
    sum_x = _ts_sum(x, w)
    sum_y = _ts_sum(y, w)
    sum_xy = _ts_sum(x * y, w)
    sum_x2 = _ts_sum(x * x, w)
    sum_y2 = _ts_sum(y * y, w)
    
    cov = (sum_xy - sum_x * sum_y / float(w)) / float(w - 1)
    var_x = (sum_x2 - sum_x * sum_x / float(w)) / float(w - 1)
    var_y = (sum_y2 - sum_y * sum_y / float(w)) / float(w - 1)
    
    std_x = torch.clamp(var_x, min=1e-6).sqrt()
    std_y = torch.clamp(var_y, min=1e-6).sqrt()
    
    return cov / (std_x * std_y)


@torch.jit.script
def _ts_res(x: torch.Tensor, y: torch.Tensor, w: int) -> torch.Tensor:
    """O(1) 复杂度的线性回归残差 MRes(window, x, y): y - (alpha + beta * x)"""
    if w <= 1:
        return torch.zeros_like(x)
    sum_x = _ts_sum(x, w)
    sum_y = _ts_sum(y, w)
    sum_xy = _ts_sum(x * y, w)
    sum_x2 = _ts_sum(x * x, w)
    
    cov = (sum_xy - sum_x * sum_y / float(w)) / float(w - 1)
    var_x = (sum_x2 - sum_x * sum_x / float(w)) / float(w - 1)
    
    beta = cov / torch.clamp(var_x, min=1e-6)
    
    mu_x = sum_x / float(w)
    mu_y = sum_y / float(w)
    alpha = mu_y - beta * mu_x
    
    return y - (alpha + beta * x)


# ── 截面算子 ─────────────
def _op_avg(x: torch.Tensor) -> torch.Tensor:
    """截面均值"""
    return x.mean(dim=0, keepdim=True).expand_as(x)


# ── 动态构建 OPS_CONFIG ─────────────
OPS_CONFIG = []

_CS_FUNCS = {
    'SIGN': (torch.sign, 1),
    'AVG': (_op_avg, 1),
    'ADDED': (lambda x, y: x + y, 2),
    'SUBBED': (lambda x, y: x - y, 2),
    'MUL': (lambda x, y: x * y, 2),
    'MOD': (torch.fmod, 2),
}
for name, (func, arity) in _CS_FUNCS.items():
    OPS_CONFIG.append((name, func, arity))

_TS_FUNCS = {
    'EMA': (lambda x, p: _ts_ema(x, p), 1),
    'RSI': (lambda x, p: _ts_rsi(x, p), 1),
    'MCORR': (lambda x, y, p: _ts_corr(x, y, p), 2),
    'MRes': (lambda x, y, p: _ts_res(x, y, p), 2),
    'MSUM': (lambda x, p: _ts_sum(x, p), 1),
    'MKURT': (lambda x, p: _ts_kurt(x, p), 1),
    'MSKEW': (lambda x, p: _ts_skew(x, p), 1),
    'MSTD': (lambda x, p: _ts_std(x, p), 1),
}

for p in ModelConfig.DEFAULT_PERIODS:
    for prefix, (func, arity) in _TS_FUNCS.items():
        name = f"{prefix}_{p}"
        if arity == 1:
            def _bind1(f, period):
                return lambda x: f(x, period)
            OPS_CONFIG.append((name, _bind1(func, p), 1))
        elif arity == 2:
            def _bind2(f, period):
                return lambda x, y: f(x, y, period)
            OPS_CONFIG.append((name, _bind2(func, p), 2))
