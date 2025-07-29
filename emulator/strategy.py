import pdb
from collections import OrderedDict
from constant import OrderStatus, OrderType, Offset, Direction
from object import OrderData


class Strategy(object):

    def __init__(self, strategy_id, params={}, session=0, at_id=10001):
        self._strategy_id = strategy_id
        self._params = params
        self._session = session

        self._is_trading = False
        self._cross_limit_order = OrderedDict()  ## 撮合队列
        self._limit_order = OrderedDict()  # 委托队列
        self._history_limit_order = OrderedDict()  # 历史委托单
        self._history_limit_turnover = OrderedDict()  # 历史成交记录
        self._working_limit_order = OrderedDict()  # 正在委托队列
        self._trader_record = OrderedDict()  # 交易记录存储
        self._long_position = OrderedDict()  # 多头持仓
        self._short_position = OrderedDict()  # 空头持仓

    def reset(self):
        self._limit_order.clear()
        self._history_limit_turnover.clear()
        self._history_limit_order.clear()

    def initialize(self, working_order, cross_limit_order=[]):  # 初始化
        self._working_limit_order = working_order

        self._cross_limit_order = cross_limit_order

    def before_market_open(self, date):  # 开盘前运行函数
        self._is_trading = True

    def after_market_close(self, date, market_daily):  # 收盘后运行函数
        self._is_trading = False

    def after_market_order(self, order):  # 处理未成交单
        raise NotImplementedError

    def on_tick(self, tick):  # 接收tick数据
        raise NotImplementedError

    def on_bar(self, bar):  ## 接收bar 数据
        raise NotImplementedError

    def on_order(self, order):  #委托通知
        if order.status == OrderStatus.ENTRUST_TRADED:  # 委托成功锁住费用
            if order.offset == Offset.OPEN:
                pass ## 账户刷新
            else:
                pass ## 账户刷新
        elif order.status == OrderStatus.REJECTED or order.status == OrderStatus.CANCELLED:
            pass ## 账户刷新

    def on_turnover(self, turnover, order):  # 成交通知
        if turnover.offset == Offset.OPEN:  # 开仓
            if turnover.direction == Direction.LONG:  # 多头开仓成交
                self._long_position[turnover.tradeid] = turnover
            elif turnover.direction == Direction.SHORT:  # 空头开仓成交
                self._short_position[turnover.tradeid] = turnover

        else:
            if turnover.direction == Direction.LONG:  # 平空头
                if order.tradeid in self._short_position:
                    v = self._short_position[order.tradeid]
                    del self._short_position[order.tradeid]

            elif turnover.direction == Direction.SHORT:  # 平多头
                pdb.set_trace()
                if order.tradeid in self._long_position:
                    v = self._long_position[order.tradeid]
                    del self._long_position[order.tradeid]

        self._history_limit_turnover[turnover.tradeid] = turnover

    def calc_result(self):  # 策略完成后进行结算
        pass

    def on_calc_settle(self, date, daily_settle_price):  # 每日结算
        pass

    def create_order(self, symbol, price, volume, order_type, direction,
                     offset, tradeid, create_time):
        order = OrderData(
            symbol=symbol,
            strategy_id=self._strategy_id,
            order_id=OrderData.create_order_id(),
            order_type=order_type,
            direction=direction,
            offset=offset,
            status=OrderStatus.NOT_TRADED,
            price=price,
            volume=volume,
            #traded=0,
            is_clear=0,
            tradeid=tradeid,
            create_time=create_time)
        self._history_limit_order[order.order_id] = order
        return order

    ## 开多仓
    def order_buy(self,
                  symbol,
                  create_time,
                  price,
                  volume=1,
                  order_type=OrderType.LIMIT):
        order = self.create_order(symbol=symbol,
                                  price=price,
                                  volume=volume,
                                  order_type=order_type,
                                  direction=Direction.LONG,
                                  offset=Offset.OPEN,
                                  tradeid="0",
                                  create_time=create_time)
        self._working_limit_order[order.order_id] = order

    ## 开空仓
    def order_short(self,
                    symbol,
                    create_time,
                    price,
                    volume,
                    order_type=OrderType.LIMIT):
        order = self.create_order(symbol=symbol,
                                  price=price,
                                  volume=volume,
                                  order_type=order_type,
                                  direction=Direction.SHORT,
                                  offset=Offset.OPEN,
                                  tradeid="0",
                                  create_time=create_time)
        self._working_limit_order[order.order_id] = order

    ## 平多仓
    def order_sell(self,
                   symbol,
                   create_time,
                   price,
                   volume,
                   tradeid,
                   order_type=OrderType.LIMIT):
        order = self.create_order(symbol=symbol,
                                  price=price,
                                  volume=volume,
                                  order_type=order_type,
                                  direction=Direction.SHORT,
                                  offset=Offset.CLOSE,
                                  tradeid=tradeid,
                                  create_time=create_time)
        self._working_limit_order[order.order_id] = order

    ## 平空仓
    def order_cover(self,
                    symbol,
                    create_time,
                    tradeid,
                    price,
                    volume,
                    order_type=OrderType.LIMIT):
        order = self.create_order(symbol=symbol,
                                  price=price,
                                  volume=volume,
                                  order_type=order_type,
                                  direction=Direction.LONG,
                                  offset=Offset.CLOSE,
                                  tradeid=tradeid,
                                  create_time=create_time)
        pdb.set_trace()
        self._working_limit_order[order.order_id] = order
