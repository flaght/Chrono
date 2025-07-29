import pandas as pd
from collections import OrderedDict
from constant import Offset, Direction

class Portfolio(object):
    """
    负责管理策略的持仓、资金和盈亏计算。
    """
    def __init__(self, initial_capital=1_000_000, multiplier=1, commission_rate={'open': 0.0, 'close': 0.0}):
        self.initial_capital = initial_capital
        self.multiplier = multiplier
        self.commission_rate = commission_rate

        self.net_value = initial_capital  # 账户净值
        self.realized_pnl = 0.0           # 已实现盈亏
        self.unrealized_pnl = 0.0         # 未实现盈亏（浮盈）
        self.total_commission = 0.0       # 累计手续费

        self.long_position = OrderedDict()  # 多头持仓
        self.short_position = OrderedDict() # 空头持仓

        self.daily_pnl = 0.0 # 当日盈亏（包括价差和手续费）
        self.daily_results = [] # 存储每日结算结果

    def on_turnover(self, turnover):
        """处理一笔成交，更新持仓和计算已实现盈亏及手续费。"""
        
        # 1. 计算并记录手续费
        notional_value = turnover.price1 * turnover.volume * self.multiplier
        commission_type = 'open' if turnover.offset == Offset.OPEN else 'close'
        rate = self.commission_rate.get(commission_type, 0.0)
        commission_cost = notional_value * rate
        
        self.total_commission += commission_cost
        
        # 2. 更新持仓和计算已实现盈亏
        realized_pnl_on_trade = 0.0
        if turnover.offset == Offset.OPEN:
            # 开仓时，将成交价作为持仓成本价（price2）
            turnover.price2 = turnover.price1
            if turnover.direction == Direction.LONG:
                self.long_position[turnover.tradeid] = turnover
            else: # SHORT
                self.short_position[turnover.tradeid] = turnover
        else: # 平仓
            if turnover.direction == Direction.LONG: # 买入平空
                if turnover.tradeid in self.short_position:
                    position = self.short_position.pop(turnover.tradeid)
                    realized_pnl_on_trade = (position.price2 - turnover.price1) * position.volume * self.multiplier
            else: # 卖出平多
                if turnover.tradeid in self.long_position:
                    position = self.long_position.pop(turnover.tradeid)
                    realized_pnl_on_trade = (turnover.price1 - position.price2) * position.volume * self.multiplier
        
        # 3. 更新账户资金
        self.realized_pnl += (realized_pnl_on_trade - commission_cost)
        self.daily_pnl += (realized_pnl_on_trade - commission_cost)
        
    def update_unrealized_pnl(self, last_price):
        """根据最新价格更新持仓的浮动盈亏。"""
        self.unrealized_pnl = 0.0
        # 多头浮盈
        for trade in self.long_position.values():
            self.unrealized_pnl += (last_price - trade.price2) * trade.volume * self.multiplier
        # 空头浮盈
        for trade in self.short_position.values():
            self.unrealized_pnl += (trade.price2 - last_price) * trade.volume * self.multiplier
        return self.unrealized_pnl
        
    def get_current_net_value(self):
        """获取当前的总资产净值。"""
        return self.initial_capital + self.realized_pnl + self.unrealized_pnl

    def mark_to_market(self, date, settle_price):
        """执行每日盯市结算。"""
        # 1. 计算当日持仓结算盈亏
        settle_pnl = 0.0
        for trade in self.long_position.values():
            settle_pnl += (settle_price - trade.price2) * trade.volume * self.multiplier
            trade.price2 = settle_price # 更新成本为结算价
        
        for trade in self.short_position.values():
            settle_pnl += (trade.price2 - settle_price) * trade.volume * self.multiplier
            trade.price2 = settle_price # 更新成本为结算价
            
        # 2. 更新当日盈亏和账户总净值
        self.daily_pnl += settle_pnl
        self.net_value += self.daily_pnl
        
        # 3. 记录当日结算数据
        self.daily_results.append({
            'date': pd.to_datetime(date),
            'pnl': self.daily_pnl,
            'net_value': self.net_value,
            'realized_pnl': self.realized_pnl,
            'commission': self.total_commission
        })
        
        # 4. 重置当日计数器
        self.daily_pnl = 0.0