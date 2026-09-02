import numpy as np
from lumina.impulse.fixed import *


def ac002(close, window, fast, slow, alpha=0.99, var_floor=1e-10, ewm=False):
    """
    DMA Adaptive Z-Score

    对分钟级 log return 特征 D_t 维护多个不同遗忘速度的在线矩估计器，
    通过贝叶斯模型平均动态合成多尺度条件均值/条件波动，
    输出当前特征相对动态基线的标准化残差。
    """
    method = 'ewm' if ewm else 'rolling'
    # alpha 为权重老化正则，0.99 对应约 199 根 K 线的 EWMA 等效跨度
    forget = min(max(float(alpha), 1e-6), 1.0 - 1e-6)
    span_alpha = max(int(round((1.0 + forget) / (1.0 - forget))), 2)

    # 默认特征：对数收益率
    D = safe_shift(close, 1)

    # 候选统计量只使用历史信息，因此把 D 再滞后一帧
    D_hist = D.shift(1)

    # 由 fast/slow 生成 6 个候选时间常数
    f = int(fast)
    s = int(slow)
    if f > s:
        f, s = s, f
    f = max(f, 2)
    s = max(s, f + 1)
    tau_list = np.rint(np.geomspace(f, s, num=6)).astype(int)

    mus = []
    vars_ = []
    log_liks = []

    for tau in tau_list:
        tau = int(tau)

        # ewm 模式下将 tau 转换为 pandas span，使指数衰减近似 exp(-1/tau)
        if method == 'ewm':
            wspan = int(round(2.0 / (1.0 - np.exp(-1.0 / tau)) - 1.0))
        else:
            wspan = tau
        wspan = max(wspan, 2)

        # 候选模型基于历史特征的条件均值与条件方差
        mu_i = roller_mean(D_hist, wspan, 1, method)
        var_i = roller_mean(D_hist * D_hist, wspan, 1, method) - mu_i * mu_i
        var_i = np.maximum(var_i, var_floor)

        mus.append(mu_i)
        vars_.append(var_i)

        # 高斯对数似然，用于后续模型权重更新
        log_norm = self_log(2.0 * np.pi * var_i)
        sq_norm = safe_div((D - mu_i) ** 2, var_i)
        log_lik_i = -0.5 * log_norm - 0.5 * sq_norm

        # 权重更新必须在输出 z_t 之后发生，因此将似然滞后一帧
        log_liks.append(log_lik_i.shift(1))

    # 模型权重：对历史对数似然做 EWMA/滚动平均，再执行 softmax 归一化
    scores = []
    for llag in log_liks:
        score_i = roller_mean(llag, span_alpha, 1, method)
        scores.append(score_i.fillna(0.0))

    max_score = scores[0]
    for score_i in scores[1:]:
        max_score = np.maximum(max_score, score_i)

    weights = []
    sum_exp = None
    for score_i in scores:
        exp_i = np.exp(score_i - max_score)
        weights.append(exp_i)
        sum_exp = exp_i if sum_exp is None else sum_exp + exp_i

    weights = [safe_div(w, sum_exp) for w in weights]

    # 模型平均预测均值
    mu_prev = None
    for w_i, mu_i in zip(weights, mus):
        contrib = w_i * mu_i
        mu_prev = contrib if mu_prev is None else mu_prev + contrib

    # 模型平均预测方差：候选内部方差 + 候选间方差
    sigma2_prev = None
    for w_i, mu_i, var_i in zip(weights, mus, vars_):
        contrib = w_i * (var_i + (mu_i - mu_prev) ** 2)
        sigma2_prev = contrib if sigma2_prev is None else sigma2_prev + contrib

    sigma2_prev = np.maximum(sigma2_prev, var_floor)

    # 动态标准化残差
    z = safe_div(D - mu_prev, np.sqrt(sigma2_prev))

    # 框架硬性要求：最终平滑
    factor = roller_mean(z, window, 1, method)

    return factor