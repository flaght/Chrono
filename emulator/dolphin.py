import pdb
import pandas as pd
# 导入基类
from strategy import Strategy
from constant import OrderStatus, Offset, Direction


class Dolphin(Strategy):

    def __init__(self, strategy_id, params={}, session=0, at_id=10001):
        super(Dolphin, self).__init__(strategy_id=strategy_id,
                                      params=params,
                                      session=session,
                                      at_id=at_id)

    def initialize(self, working_order, cross_limit_order):
        super(Dolphin, self).initialize(working_order, cross_limit_order)
        self._bar_list = []
        self._count = 40
        self._long_position_count = 0
        self._short_position_count = 0
        self._long_target = 1
        self._short_target = 1

    # on_tick, on_bar_logic, on_order, on_turnover 等方法几乎不需要改变
    # 因为父类已经处理了 portfolio 的交互，并且通过 @property 暴露了持仓
    # 子类可以像以前一样访问 self._long_position 和 self._short_position

    def on_tick(self, tick):
        if len(self._bar_list) < self._count:
            return

        ## 开多头，平空头
        if tick.last_price > self.upper_band:
            # self._short_position 现在通过 property 访问 portfolio.short_position
            for tid, position in list(self._short_position.items()):
                if self._short_position_count > 0:
                    self.order_cover(symbol=tick.symbol,
                                     create_time=tick.create_time,
                                     price=tick.last_price,
                                     tradeid=position.tradeid,
                                     volume=position.volume)
                    self._short_position_count -= 1
            while self._long_target > 0:
                self.order_buy(symbol=tick.symbol,
                               create_time=tick.create_time,
                               price=tick.last_price,
                               volume=1)
                self._long_target -= 1

        ## 开空头，平多头
        elif tick.last_price < self.down_band:
            for tid, position in list(self._long_position.items()):
                if self._long_position_count > 0:
                    self.order_sell(symbol=tick.symbol,
                                    create_time=tick.create_time,
                                    price=tick.last_price,
                                    tradeid=position.tradeid,
                                    volume=position.volume)
                    self._long_position_count -= 1
            while self._short_target > 0:
                self.order_short(symbol=tick.symbol,
                                 create_time=tick.create_time,
                                 price=tick.last_price,
                                 volume=1)
                self._short_target -= 1

    def on_bar_logic(self, bar):
        market = {
            'THIGH': bar.high_price,
            'TLOW': bar.low_price,
            'TCLOSE': bar.close_price,
            'TOPEN': bar.open_price,
            'TTIME': bar.create_time
        }
        self._bar_list.append(market)

        if len(self._bar_list) < self._count: return
        if len(self._bar_list) > self._count: self._bar_list.pop(0)

        df = pd.DataFrame(self._bar_list)
        self.mean = df['TCLOSE'].mean()
        self.std = df['TCLOSE'].std()
        self.upper_band = self.mean + 2 * self.std
        self.down_band = self.mean - 2 * self.std

    def on_turnover(self, turnover, order):
        # 必须调用父类方法来触发 portfolio 更新
        super(Dolphin, self).on_turnover(turnover, order)
        print(self._long_target, self._short_target)
        if turnover.direction == Direction.LONG and turnover.offset == Offset.OPEN:  ## 开多头
            self._long_position_count += 1
        elif turnover.direction == Direction.SHORT and turnover.offset == Offset.OPEN:  ## 开空头
            self._short_position_count += 1
        elif turnover.direction == Direction.SHORT and turnover.offset == Offset.CLOSE:  ## 平多头
            self._long_target += 1
        elif turnover.direction == Direction.LONG and turnover.offset == Offset.CLOSE:  ## 平空头
            self._short_target += 1

    def on_order(self, order):
        super(Dolphin, self).on_order(order=order)
        if order.status == OrderStatus.REJECTED:
            if order.offset == Offset.OPEN and order.direction == Direction.LONG:
                self._long_target += 1
            elif order.offset == Offset.OPEN and order.direction == Direction.SHORT:
                self._short_target += 1
            elif order.offset == Offset.CLOSE and order.direction == Direction.LONG:
                self._short_position_count += 1
            elif order.offset == Offset.CLOSE and order.direction == Direction.SHORT:
                self._long_position_count += 1
