import pdb
import pandas as pd
from strategy import Strategy
from constant import OrderStatus, Offset, Direction


class Dolphin(Strategy):

    def __init__(self, strategy_id, params={}, session=0, at_id=10001):
        # 关键：调用父类的构造函数，传递所有参数
        super(Dolphin, self).__init__(strategy_id=strategy_id,
                                      params=params,
                                      session=session,
                                      at_id=at_id)

    def initialize(self, working_order, cross_limit_order):  # 初始化
        super(Dolphin, self).initialize(working_order, cross_limit_order)
        self._bar_list = []
        self._count = 40
        self._long_position_count = 0
        self._short_position_count = 0
        self._long_target = 1
        self._short_target = 1

    def before_market_open(self, date):
        super(Dolphin, self).before_market_open(date=date)

    def after_market_close(self, date, market_daily):
        # 确保调用父类方法来记录每日结果
        return super().after_market_close(date, market_daily)

    def after_market_order(self, order):
        pass

    def on_tick(self, tick):
        if len(self._bar_list) < self._count:
            return
        ## 穿上轨 开多仓，平空仓
        if tick.last_price > self.upper_band:
            # 平空仓时，需要传递正确的 tradeid
            for tid, position in list(self._short_position.items()): # 使用list()避免在迭代时修改字典
                if self._short_position_count > 0:
                    self.order_cover(symbol=tick.symbol,
                                     create_time=tick.create_time,
                                     price=tick.last_price,
                                     tradeid=position.tradeid, # 修正：传递正确的tradeid
                                     volume=position.volume)
                    self._short_position_count -= 1

            while self._long_target > 0:
                self.order_buy(symbol=tick.symbol,
                               create_time=tick.create_time,
                               price=tick.last_price,
                               volume=1)
                self._long_target -= 1

        ## 穿下轨 开空仓，平多仓
        elif tick.last_price < self.down_band:
            for tid, position in list(self._long_position.items()): # 使用list()避免在迭代时修改字典
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
    
    # 将原on_bar的逻辑放入on_bar_logic
    def on_bar_logic(self, bar):
        market = {
            'THIGH': bar.high_price,
            'TLOW': bar.low_price,
            'TCLOSE': bar.close_price,
            'TOPEN': bar.open_price,
            'TTIME': bar.create_time
        }
        self._bar_list.append(market)

        if len(self._bar_list) < self._count:
            return
        elif len(self._bar_list) > self._count:
            self._bar_list.pop(0)

        ## 计算布林带
        df = pd.DataFrame(self._bar_list)
        self.mean = df['TCLOSE'].mean()
        self.std = df['TCLOSE'].std()
        self.upper_band = self.mean + 2 * self.std
        self.down_band = self.mean - 2 * self.std

    def on_order(self, order):  # 委托通知
        super(Dolphin, self).on_order(order=order)
        if order.status == OrderStatus.REJECTED:  ## 拒绝单 ## 回填参数
            if order.offset == Offset.OPEN and order.direction == Direction.LONG:
                self._long_target += 1
            elif order.offset == Offset.OPEN and order.direction == Direction.SHORT:
                self._short_target += 1

            elif order.offset == Offset.CLOSE and order.direction == Direction.LONG:
                self._short_position_count += 1
            elif order.offset == Offset.CLOSE and order.direction == Direction.SHORT:
                self._long_position_count += 1

    def on_turnover(self, turnover, order):
        # 确保调用父类方法来处理持仓和计算已实现盈亏
        super(Dolphin, self).on_turnover(turnover=turnover, order=order)
        ## 开多仓
        if turnover.direction == Direction.LONG and turnover.offset == Offset.OPEN:
            self._long_position_count += 1
        ## 开空仓
        elif turnover.direction == Direction.SHORT and turnover.offset == Offset.OPEN:
            self._short_position_count += 1