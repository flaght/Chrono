import pdb
import numpy as np
import pandas as pd
from collections import OrderedDict
from constant import OrderStatus, OrderType, Offset, Direction
from object import OrderData


class Strategy(object):

    def __init__(self, strategy_id, params={}, session=0, at_id=10001):
        self._strategy_id = strategy_id
        self._params = params
        self.initial_capital = params.get('initial_capital', 1_000_000)
        self.multiplier = params.get('multiplier', 1)

        # --- 新增：获取手续费率配置 ---
        # 允许为开仓和平仓设置不同费率，默认为0
        self.commission_rate = params.get('commission_rate', {
            'open': 0.0,
            'close': 0.0
        })

        self._session = session
        self._is_trading = False

        self._cross_limit_order = OrderedDict()
        self._limit_order = OrderedDict()
        self._history_limit_order = OrderedDict()
        self._history_limit_turnover = OrderedDict()
        self._working_limit_order = OrderedDict()

        self._long_position = OrderedDict()
        self._short_position = OrderedDict()

        self.net_value = self.initial_capital
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.current_day_pnl = 0.0

        # --- 新增：追踪总手续费 ---
        self.total_commission = 0.0

        self.net_value_series = []
        self.daily_results = []

    def reset(self):
        self._limit_order.clear()
        self._history_limit_turnover.clear()
        self._history_limit_order.clear()

    def initialize(self, working_order, cross_limit_order=[]):
        self._working_limit_order = working_order
        self._cross_limit_order = cross_limit_order
        if not self.net_value_series:
            self.net_value_series.append({
                'time': None,
                'net_value': self.initial_capital
            })

    def before_market_open(self, date):
        self._is_trading = True
        self.current_day_pnl = 0.0

    def after_market_close(self, date, market_daily):
        self._is_trading = False
        self.daily_results.append({
            'date': pd.to_datetime(date),
            'pnl': self.current_day_pnl,
            'net_value': self.net_value,
            'realized_pnl': self.realized_pnl,
            'commission': self.total_commission  # 记录总手续费
        })

    def after_market_order(self, order):
        raise NotImplementedError

    def on_tick(self, tick):
        raise NotImplementedError

    def on_bar(self, bar):
        self.unrealized_pnl = 0.0
        for trade in self._long_position.values():
            self.unrealized_pnl += (bar.close_price - trade.price2
                                    ) * trade.volume * self.multiplier
        for trade in self._short_position.values():
            self.unrealized_pnl += (trade.price2 - bar.close_price
                                    ) * trade.volume * self.multiplier

        # --- 修改：净值计算现在也间接包含了手续费的影响 ---
        # self.realized_pnl 已经被扣除了手续费
        current_net_value = self.initial_capital + self.realized_pnl + self.unrealized_pnl

        self.net_value_series.append({
            'time': bar.create_time,
            'net_value': current_net_value
        })

        self.on_bar_logic(bar)

    def on_bar_logic(self, bar):
        raise NotImplementedError

    def on_order(self, order):
        if order.status == OrderStatus.ENTRUST_TRADED:
            pass
        elif order.status == OrderStatus.REJECTED or order.status == OrderStatus.CANCELLED:
            pass

    def on_turnover(self, turnover, order):
        # --- 新增：计算并扣除本次交易的手续费 ---
        notional_value = turnover.price1 * turnover.volume * self.multiplier
        commission_type = 'open' if turnover.offset == Offset.OPEN else 'close'
        rate = self.commission_rate.get(commission_type, 0.0)
        commission_cost = notional_value * rate

        # 更新总手续费，并从当日和累计已实现盈亏中扣除
        self.total_commission += commission_cost
        self.realized_pnl -= commission_cost
        self.current_day_pnl -= commission_cost

        # --- 以下逻辑不变，但pnl计算现在是纯粹的价差盈亏 ---
        if turnover.offset == Offset.OPEN:
            turnover.price2 = turnover.price1
            if turnover.direction == Direction.LONG:
                self._long_position[turnover.tradeid] = turnover
            elif turnover.direction == Direction.SHORT:
                self._short_position[turnover.tradeid] = turnover
        else:
            pnl = 0.0
            if turnover.direction == Direction.LONG:
                if order.tradeid in self._short_position:
                    position = self._short_position.pop(order.tradeid)
                    pnl = (position.price2 -
                           turnover.price1) * position.volume * self.multiplier
            elif turnover.direction == Direction.SHORT:
                if order.tradeid in self._long_position:
                    position = self._long_position.pop(order.tradeid)
                    pnl = (turnover.price1 -
                           position.price2) * position.volume * self.multiplier

            # 将价差盈亏计入账户
            self.realized_pnl += pnl
            self.current_day_pnl += pnl

        self._history_limit_turnover[turnover.tradeid] = turnover

    def on_calc_settle(self, date, daily_settle_price):
        settle_pnl = 0.0
        for trade in self._long_position.values():
            pnl = (daily_settle_price -
                   trade.price2) * trade.volume * self.multiplier
            settle_pnl += pnl
            trade.price2 = daily_settle_price
        for trade in self._short_position.values():
            pnl = (trade.price2 -
                   daily_settle_price) * trade.volume * self.multiplier
            settle_pnl += pnl
            trade.price2 = daily_settle_price

        self.current_day_pnl += settle_pnl
        self.net_value += self.current_day_pnl

    def calc_result(self):
        print("\n--- Strategy Performance Analysis (with fees) ---")
        if not self.daily_results:
            print("No trades were made. No performance to analyze.")
            return

        daily_df = pd.DataFrame(self.daily_results)
        daily_df.set_index('date', inplace=True)

        daily_df['daily_return'] = daily_df['net_value'].pct_change().fillna(0)
        daily_df['cumulative_return'] = (
            1 + daily_df['daily_return']).cumprod() - 1

        total_return = daily_df['cumulative_return'].iloc[-1]
        annualized_return = total_return * (252 / len(daily_df))
        sharpe_ratio = (daily_df['daily_return'].mean() /
                        daily_df['daily_return'].std()) * np.sqrt(
                            252) if daily_df['daily_return'].std() != 0 else 0

        daily_df['cumulative_max'] = daily_df['net_value'].cummax()
        daily_df['drawdown'] = daily_df['net_value'] / daily_df[
            'cumulative_max'] - 1
        max_drawdown = daily_df['drawdown'].min()

        print(f"Initial Capital: {self.initial_capital:,.2f}")
        print(f"Final Net Value: {self.net_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        # --- 新增：打印总手续费 ---
        print(f"Total Commission Paid: {self.total_commission:,.2f}")

        try:
            import matplotlib.pyplot as plt
            daily_df['net_value'].plot(title='Portfolio Net Value Over Time',
                                       grid=True)
            plt.ylabel('Net Value')
            plt.xlabel('Date')
            plt.show()
        except ImportError:
            print("\nMatplotlib not installed. Skipping plot generation.")

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
