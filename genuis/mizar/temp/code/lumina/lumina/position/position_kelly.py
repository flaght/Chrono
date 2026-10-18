# -*- encoding:utf-8 -*-
from lumina.position.base import PositionBase


class KellyPosition(PositionBase):
    def fit_weight(self, factor_object):
        # 败率
        loss_rate = 1 - self.win_rate
        # kelly 计算出仓位比例
        kelly_pos = self.win_rate - loss_rate / (self.gains_mean / self.losses_mean)
        # 最大仓位限制，依然受上层最大仓位控制限制，eg：如果kelly计算出全仓，依然会减少到75%，如修改需要修改最大仓位值
        kelly_pos = self.pos_max if kelly_pos > self.pos_max else kelly_pos
        return kelly_pos

    def fit_position(self, factor_object):
        pos = self.fit_weight(factor_object)
        return self.read_cash * pos / self.bp * self.deposit_rate
    
    def _init_self(self, **kwargs):
        """kelly仓位控制管理类初始化设置"""
        # 默认kelly 仓位胜率 0.50
        self.win_rate = kwargs.pop('win_rate', 0.50)
        # 默认平均获利期望0.10
        self.gains_mean = kwargs.pop('gains_mean', 0.10)
        # 默认平均亏损期望0.05
        self.losses_mean = kwargs.pop('losses_mean', 0.05)

        """以默认的设置kelly根据计算0.5 - 0.5 / (0.10 / 0.05) 仓位将是0.25即25%"""
