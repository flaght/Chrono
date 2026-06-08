import numpy as np
from ultron.utilities.logger import kd_logger


class Adaptive(object):
    """
    一个自适应调整相关性惩罚系数alpha的类。
    它通过维持一个目标惩罚比率，来动态调整alpha，使其与基础评估函数的尺度解耦。
    """

    def __init__(self,
                 initial_alpha=0.05,
                 target_penalty_ratio=0.5,
                 adjustment_speed=0.1,
                 lookback_period=10,
                 alpha_min=0.1,
                 alpha_max=3.0):
        """
        初始化自适应Alpha调节器。

        :param initial_alpha: float, 初始的alpha值。
        :param target_penalty_ratio: float, 目标惩罚比率。例如，0.5意味着我们希望惩罚项的量级
                                     大约是基础表现得分的一半。这是核心控制参数。
        :param adjustment_speed: float, 调整速度，控制每次更新的步长 (0 < speed <= 1)。
        :param lookback_period: int, 用于计算滑动平均的历史窗口大小。
        """
        self._alpha = initial_alpha
        self._target_penalty_ratio = target_penalty_ratio
        self._adjustment_speed = adjustment_speed
        self._lookback_period = lookback_period

        # 用于存储历史数据的列表
        self._history_base_performance = []
        self._history_max_corr = []

        self._alpha_min = alpha_min
        self._alpha_max = alpha_max

    @property
    def alpha(self):
        return self._alpha

    def update(self, base_performance: float, max_corr: float):
        """
        根据最新一代的表现来更新alpha。

        :param base_performance: float, 最新一次评估的基础表现分 (e.g., abs(IC), ICIR)。
        :param max_corr: float, 最新一次评估的最大相关系数。
        """
        if base_performance <= 0 or max_corr <= 1e-6:
            # 如果表现不佳或相关性极低，则不进行调整，避免除零或无效更新
            kd_logger.info("very bad base_performance:{0} or max_corr:{1}".format(
                base_performance, max_corr))
            return

        # 记录历史数据
        self._history_base_performance.append(base_performance)
        self._history_max_corr.append(max_corr)

        # 保持历史窗口大小
        if len(self._history_base_performance) > self._lookback_period:
            self._history_base_performance.pop(0)
            self._history_max_corr.pop(0)

        # 1. 计算历史滑动平均值，以获得更平滑的估计
        avg_base_performance = np.mean(self._history_base_performance)
        avg_max_corr = np.mean(self._history_max_corr)

        if avg_base_performance <= 0 or avg_max_corr <= 1e-6:
            kd_logger.info(
                "very bad avg_base_performance:{0} or avg_max_corr:{1}".format(
                    avg_base_performance, avg_max_corr))

        # 2. 计算“理想”的alpha值
        # 这个理想值能使得在平均情况下，惩罚项达到目标比率
        ideal_alpha = (self._target_penalty_ratio *
                       avg_base_performance) / avg_max_corr

        # 3. 平滑地向理想alpha值调整
        # 使用指数移动平均(EMA)的思想，防止alpha剧烈波动
        old_alpha = self._alpha
        self._alpha = (1 - self._adjustment_speed
                       ) * self._alpha + self._adjustment_speed * ideal_alpha
        kd_logger.info("update alpha: {0}===>{1}".format(old_alpha, self._alpha))
