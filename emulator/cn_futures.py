import pdb
from collections import OrderedDict
from loader import Loader
from object import BarData, TradeData
from constant import Interval, OrderStatus, OrderType, Direction, Offset


class CNFutures(object):

    def __init__(self, code, uri, strategies_pool={}):
        self._code = code
        self.strategies_pool = strategies_pool
        self.loader = Loader.create_loader(uri=uri)
        self.init_order()

        ## 制作1分钟bar
        self.market_bar = None
        self.bar_mintue = None

    def init_order(self):
        self.limit_order = OrderedDict()  # 限制价单
        self.cannel_order = OrderedDict()  # 撤销单
        self.turnover_order = OrderedDict()  # 成交单
        self.working_limit_order = OrderedDict()  # 正在委托单
        self.working_limit_turnover = OrderedDict()  # 持仓
        self.recovery_limit_order = OrderedDict()  # 回收队列，用于下次撮合
        for name, strategy in self.strategies_pool.items():
            strategy.initialize(working_order=self.working_limit_order,
                                cross_limit_order=[])

    def _recovery_to_working(self):
        for oid, order in self.recovery_limit_order.items():
            self.working_limit_order[oid] = order
        self.recovery_limit_order.clear()

    ## 加载器适配，即可以文件也可以ddb
    def load_tick(self, trade_date):
        market_tick_list = self.loader.fetch_ticket(code=self._code,
                                                    trade_date=trade_date)
        return market_tick_list

    def load_daily(self, trade_date):
        market_daily = self.loader.fetch_daily(code=self._code,
                                               trade_date=trade_date)
        return market_daily

    def create_bar(self, market_tick):
        tick_mintue = int(market_tick.create_time.minute)
        if tick_mintue != self.bar_mintue:
            if self.market_bar:
                ## 刷新时间
                self.market_bar.create_time = self.market_bar.create_time.floor(
                    'min')
                for _, strategy in self.strategies_pool.items():
                    strategy.on_bar(self.market_bar)

            market_bar = BarData(code=market_tick.code,
                                 symbol=market_tick.symbol,
                                 exchange=market_tick.exchange,
                                 create_time=market_tick.create_time,
                                 interval=Interval.MINUTE,
                                 volume=market_tick.volume,
                                 turnover=market_tick.turnover,
                                 open_interest=market_tick.open_interest,
                                 open_price=market_tick.last_price,
                                 high_price=market_tick.last_price,
                                 low_price=market_tick.last_price,
                                 close_price=market_tick.last_price)
            self.bar_mintue = tick_mintue
            self.market_bar = market_bar
        else:
            market_bar = self.market_bar
            market_bar.high_price = max(market_bar.high_price,
                                        market_tick.last_price)
            market_bar.low_price = min(market_bar.low_price,
                                       market_tick.last_price)
            market_bar.close_price = market_tick.last_price
            market_bar.volume = market_tick.volume
            market_bar.open_interest = market_tick.open_interest
            market_bar.turnover = market_tick.turnover
            market_bar.create_time = market_tick.create_time

    def _cross_limit_order(self, market_tick):
        buy_cross_price = market_tick.ask_price_1 if market_tick.ask_price_1 > 0 else market_tick.last_price
        sell_cross_price = market_tick.bid_price_1 if market_tick.bid_price_1 > 0 else market_tick.last_price
        buy_best_price = market_tick.ask_price_1 if market_tick.ask_price_1 > 0 else market_tick.last_price
        sell_best_price = market_tick.bid_price_1 if market_tick.bid_price_1 > 0 else market_tick.last_price
        for oid, order in self.working_limit_order.items():
            if order.status == OrderStatus.NOT_TRADED:
                order.status = OrderStatus.ENTRUST_TRADED  # 修正为赋值
                # 委托成功推送信息给策略
                strategy = self.strategies_pool[order.strategy_id]
                strategy.on_order(order)
                # 判断是否会成交
                # 市价单和限价单判断
                # 若对手价不存在，则改用市价单
                buy_cross = False
                sell_cross = False
                if order.order_type == OrderType.MARKET or buy_cross_price == 0.0 or sell_cross_price == 0.0:
                    if order.direction == Direction.LONG:
                        buy_cross = True
                        sell_cross = False
                    elif order.direction == Direction.SHORT:
                        sell_cross = True
                        buy_cross = False
                    order.order_type = OrderType.MARKET  # 修正为赋值
                elif order.order_type == OrderType.LIMIT:
                    buy_cross = (order.direction == Direction.LONG
                                 and order.price >= buy_cross_price > 0)
                    sell_cross = (order.direction == Direction.SHORT
                                  and order.price <= sell_cross_price
                                  and sell_cross_price > 0)
                if buy_cross or sell_cross:
                    order.status = OrderStatus.ALL_TRADED
                    strategy = self.strategies_pool[order.strategy_id]
                    strategy.on_order(order)
                    turnover = TradeData(symbol=order.symbol,
                                         orderid=order.order_id,
                                         strategy_id=order.strategy_id,
                                         tradeid=TradeData.create_trade_id(),
                                         direction=order.direction,
                                         offset=order.offset,
                                         volume=order.volume,
                                         create_time=market_tick.create_time)
                    if buy_cross:
                        if order.order_type == OrderType.LIMIT:
                            turnover.price1 = min(market_tick.last_price, buy_best_price)
                        else:
                            turnover.price1 = market_tick.last_price
                    else:
                        if order.order_type == OrderType.LIMIT:
                            turnover.price1 = max(market_tick.last_price, sell_best_price)
                        else:
                            turnover.price1 = market_tick.last_price
                    if order.offset == Offset.OPEN:  # 开仓
                        self.working_limit_turnover[
                            turnover.tradeid] = turnover
                    else:
                        if order.tradeid in self.working_limit_turnover:
                            del self.working_limit_turnover[order.tradeid]
                    self.turnover_order[turnover.tradeid] = turnover
                    strategy.on_turnover(turnover, order)
                else:  # 无法撮合
                    if order.is_clear == 0:  #回收委托队列
                        order.status = OrderStatus.CANCELLED
                        strategy = self.strategies_pool[order.strategy_id]
                        strategy.on_order(order)
                        order.status = OrderStatus.NOT_TRADED
                        self.recovery_limit_order[order.order_id] = order
                    else:
                        order.status = OrderStatus.REJECTED
                        strategy.on_order(order)
        self.working_limit_order.clear()

    def on_tick(self, trade_date, market_tick):
        self._recovery_to_working()
        self._cross_limit_order(market_tick)

        # 行情推送给各个策略
        for name, strategy in self.strategies_pool.items():
            strategy.on_tick(market_tick)

        self.market_tick = market_tick
        self.create_bar(market_tick=market_tick)

    def start(self, trade_date):
        market_tick_list = self.load_tick(trade_date=trade_date)
        market_daily = self.load_daily(trade_date=trade_date)
        for market_tick in market_tick_list:
            if market_tick.volume == 0:
                print("!!!! {0} -{1} error".format(market_tick.create_time,
                                                   market_tick.volume))
                continue

            if market_tick.last_price >= market_tick.limit_up or market_tick.last_price <= market_tick.limit_down:
                print(
                    '!!!!!!latest_price ERROR trade_date:%s price:%f !!!!!!!' %
                    (str(trade_date), market_tick.last_price))
                continue

            self.on_tick(trade_date=trade_date, market_tick=market_tick)
            last_market_tick = market_tick
            del market_tick

        ## 通知未成交订单
        for oid, order in self.recovery_limit_order.items():
            order.status = OrderStatus.REJECTED
            strategy = self.strategies_pool[order.strategy_id]
            strategy.on_order(order)
            del self.recovery_limit_order[oid]
            del order

        # 结算当天交易及盘后处理
        for name, strategy in self.strategies_pool.items():
            strategy.after_market_close(trade_date, market_daily)
            strategy.on_calc_settle(trade_date, market_daily.settle_price)

        self.turnover_order.clear()
        self.working_limit_order.clear()
        pdb.set_trace()
        print('-->')
