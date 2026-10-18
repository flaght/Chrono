import numpy as np


class Weights(object):
    """
    根据种群的统计特性，自适应地调整ICIR和mean_ic之间的权重w1。
    """

    def __init__(self,
                 initial_w1=0.7,
                 adjustment_speed=0.1,
                 clip_range=(0.5, 0.9)):
        """
        :param initial_w1: float, 初始的w1值。
        :param adjustment_speed: float, 调整速度 (EMA平滑系数)。
        :param clip_range: tuple, w1值的上下限，防止极端情况。
        """
        self.w1 = initial_w1
        self._adjustment_speed = adjustment_speed
        self._clip_range = clip_range

    def update(self, population):
        """
        根据当前种群状态更新w1。

        :param population: 当前代的所有个体（program）列表。
        """

        abs_icir_list = [
            abs(p.icir) for p in population
            if hasattr(p, 'icir') and p.icir is not None
        ]
        abs_mean_ic_list = [
            abs(p.mean_ic) for p in population
            if hasattr(p, 'mean_ic') and p.mean_ic is not None
        ]

        # 如果数据不足，不更新
        if len(abs_icir_list) < 10 or len(abs_mean_ic_list) < 10:
            return self.w1

        # 计算变异系数 (CV)
        cv_icir = np.std(abs_icir_list) / (np.mean(abs_icir_list) + 1e-8)
        cv_mean_ic = np.std(abs_mean_ic_list) / (np.mean(abs_mean_ic_list) +
                                                 1e-8)

        # 根据相对波动性计算目标w1
        target_w1 = cv_icir / (cv_icir + cv_mean_ic + 1e-8)

        # EMA平滑更新
        self.w1 = (1 - self._adjustment_speed
                   ) * self.w1 + self._adjustment_speed * target_w1

        # 裁剪到预设范围
        self.w1 = np.clip(self.w1, self._clip_range[0], self._clip_range[1])

        return self.w1
