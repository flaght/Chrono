import pdb
from collections import OrderedDict
from constant import OrderStatus, OrderType, Offset, Direction
from object import OrderData
# 导入新模块
from portfolio import Portfolio
from metrics import Metrics


class Strategy(object):

    def __init__(self, strategy_id, params={}, session=0, at_id=10001):
        self._strategy_id = strategy_id
        self._params = params
        self._session = session
        self._is_trading = False

        # --- 核心修改：使用Portfolio对象来管理账户和持仓 ---
        self.portfolio = Portfolio(
            initial_capital=params.get('initial_capital', 1_000_000),
            multiplier=params.get('multiplier', 1),
            commission_rate=params.get('commission_rate', {}))

        self._cross_limit_order = OrderedDict()
        self._limit_order = OrderedDict()
        self._history_limit_order = OrderedDict()
        self._history_limit_turnover = OrderedDict()
        self._working_limit_order = OrderedDict()

        self.net_value_series = []  # Bar级别净值序列保持不变，用于盘中监控

    def initialize(self, working_order, cross_limit_order=[]):
        self._working_limit_order = working_order
        self._cross_limit_order = cross_limit_order
        if not self.net_value_series:
            self.net_value_series.append({
                'time':
                None,
                'net_value':
                self.portfolio.initial_capital
            })

    def before_market_open(self, date):
        self._is_trading = True

    def after_market_close(self, date, market_daily):
        self._is_trading = False
        # 委托Portfolio执行日结
        self.portfolio.mark_to_market(date, market_daily.settle_price)

    def on_tick(self, tick):
        raise NotImplementedError

    def on_bar(self, bar):
        # 委托Portfolio更新浮动盈亏
        self.portfolio.update_unrealized_pnl(bar.close_price)

        # 获取当前净值并记录
        current_net_value = self.portfolio.get_current_net_value()
        self.net_value_series.append({
            'time': bar.create_time,
            'net_value': current_net_value
        })

    def on_order(self, order):
        pass  # 订单状态变化的逻辑现在主要影响交易引擎

    def on_turnover(self, turnover, order):
        # 委托Portfolio处理成交事件
        self.portfolio.on_turnover(turnover)
        self._history_limit_turnover[turnover.tradeid] = turnover

    def on_calc_settle(self, date, daily_settle_price):
        # 注意：这个方法的逻辑已经被合并到 after_market_close -> portfolio.mark_to_market 中
        # 为了保持接口兼容，可以保留为空
        pass

    def calc_result(self):
        # 委托Metrics进行绩效分析
        calculator = Metrics(self.portfolio.daily_results,
                             self.portfolio.initial_capital)
        calculator.print_summary()

    # --- 下单函数需要访问 portfolio 中的持仓 ---
    @property
    def _long_position(self):
        return self.portfolio.long_position

    @property
    def _short_position(self):
        return self.portfolio.short_position

    # --- 下单函数保持不变 ---
    def create_order(self, symbol, price, volume, order_type, direction,
                     offset, tradeid, create_time):
        order = OrderData(symbol=symbol,
                          strategy_id=self._strategy_id,
                          order_id=OrderData.create_order_id(),
                          order_type=order_type,
                          direction=direction,
                          offset=offset,
                          status=OrderStatus.NOT_TRADED,
                          price=price,
                          volume=volume,
                          is_clear=0,
                          tradeid=tradeid,
                          create_time=create_time)
        self._history_limit_order[order.order_id] = order
        return order

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
        self._working_limit_order[order.order_id] = order
