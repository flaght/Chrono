import numpy as np
from ultron.utilities.logger import kd_logger


class Thresholdor(object):
    """
    根据种群适应度的分布，动态调整精英筛选阈值 (standard_score)。
    """

    def __init__(self,
                 initial_threshold: float,
                 target_percentile: float = 0.75,
                 min_threshold: float = 0.01,
                 max_threshold: float = 0.8,
                 adjustment_speed: float = 0.1):
        """
        初始化动态阈值调节器。

        :param initial_threshold: float, 初始的筛选阈值。
        :param target_percentile: float, 目标分位数 (0-1之间)。例如0.75意味着阈值将趋向于
                                     种群中排名前25%的个体的适应度水平。
        :param min_threshold: float, 阈值的下限，防止初期标准过低。
        :param max_threshold: float, 阈值的上限，防止标准高到不切实际。
        :param adjustment_speed: float, 调整速度 (EMA平滑系数)。
        """
        self.current_threshold = initial_threshold
        self._target_percentile = target_percentile
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._adjustment_speed = adjustment_speed

    def threshold(self):
        return self.current_threshold

    def update(self, population_fitness):
        """
        根据当前种群的适应度列表来更新阈值。

        :param population_fitness: list, 当前有效种群所有个体的适应度分数列表。
        """
        # 1. 计算目标阈值：即适应度分布的目标分位数
        # 只考虑正的fitness值，因为负值通常代表无效或极差的因子
        positive_fitness = [f for f in population_fitness if f > 0]
        if not positive_fitness:
            kd_logger.info("not positive fitness")
            return

        if not population_fitness or len(population_fitness) < 10:
            # 如果种群太小，不进行更新
            kd_logger.info("population small")
            return

        target_threshold = np.percentile(positive_fitness,
                                         self._target_percentile * 100)

        # 2. 平滑地向目标阈值调整
        old_threshold = self.current_threshold
        self.current_threshold = (1 - self._adjustment_speed) * self.current_threshold + \
                                 self._adjustment_speed * target_threshold

        # 3. 应用上下限进行裁剪
        self.current_threshold = np.clip(self.current_threshold,
                                         self._min_threshold,
                                         self._max_threshold)

        kd_logger.info(
            f"Dynamic threshold updated: {old_threshold:.4f} ===> {self.current_threshold:.4f} (target: {target_threshold:.4f})"
        )
