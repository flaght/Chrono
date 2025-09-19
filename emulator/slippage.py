import random
from constant import Slippage as SlippageEnum


class Slippage(object):

    @classmethod
    def calc(cls, types, **params):
        if SlippageEnum.FIXED == types:
            return cls.fixed(**params)
        elif SlippageEnum.PERCENTAGE == types:
            return cls.percentage(**params)
        elif SlippageEnum.RANDOM == types:
            return cls.random(**params)
        elif SlippageEnum.VOLUME == types:
            return cls.volume(**params)

    @classmethod
    def fixed(cls, **params):
        return params['fixed_value']

    @classmethod
    def percentage(cls, **params):
        slippage_rate = params['slippage_rate']
        base_price = params['base_price']
        return base_price * slippage_rate

    @classmethod
    def random(cls, **params):
        slippage_max_ticks = params['slippage_max_ticks']
        tick_size = params['tick_size']
        random_ticks = random.randint(0, slippage_max_ticks)
        return random_ticks * tick_size

    @classmethod
    def volume(cls, **params):
        ask_volume_1 = params['ask_volume_1']
        bid_volume_1 = params['bid_volume_1']
        buy_cross = params['buy_cross']
        order_volume = params['order_volume']
        base_price = params['base_price']
        slippage_volume_factor = params['slippage_volume_factor']

        market_volume = ask_volume_1 if buy_cross else bid_volume_1
        if market_volume > 0:
            # 订单量超过盘口量的比例，作为滑点影响因子
            volume_ratio = order_volume / market_volume
            # 简单线性模型：滑点与超额比例和价格正相关
            slippage_amount = base_price * volume_ratio * slippage_volume_factor
        else:  # 盘口无挂单量，可能产生极大滑点，这里简化为0
            slippage_amount = 0.0

        return slippage_amount
