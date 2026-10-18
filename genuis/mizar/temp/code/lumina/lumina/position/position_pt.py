# -*- encoding:utf-8 -*-
from lumina.position.base import PositionBase
from scipy import stats

## 适配均值回复策略
class PtPosition(PositionBase):

    def fit_weight(self, factor_object):
        # self.kl_pd_buy为买入当天的数据，获取之前的past_day_cnt天数据
        last_kl = factor_object.past_today_kl(self.kl_pd_buy, self.past_day_cnt)
        if last_kl is None or last_kl.empty:
            precent_pos = self.pos_base
        else:
            # 使用percentileofscore计算买入价格在过去的past_day_cnt天的价格位置
            precent_pos = stats.percentileofscore(last_kl.close, self.bp)
            precent_pos = (1 + (self.mid_precent - precent_pos) / 100) * self.pos_base
        # 最大仓位限制，依然受上层最大仓位控制限制，eg：如果算出全仓，依然会减少到75%，如修改需要修改最大仓位值
        precent_pos = self.pos_max if precent_pos > self.pos_max else precent_pos

    def fit_position(self, factor_object):
        pos = self.fit_weight(factor_object)
        return self.read_cash * pos / self.bp * self.deposit_rate
    
    def _init_self(self, **kwargs):
        """价格位置仓位控制管理类初始化设置"""
        # 默认平均仓位比例0.10，即10%
        self.pos_base = kwargs.pop('pos_base', 0.10)
        # 默认获取之前金融时间序列的长短数量
        self.past_day_cnt = kwargs.pop('past_day_cnt', 20)
        # 默认的比例中值，一般不需要设置
        self.mid_precent = kwargs.pop('mid_precent', 50.0)