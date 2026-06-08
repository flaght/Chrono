import pdb
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from ultron.tradingday import advanceDateByCalendar


def _to_time_str(value) -> str:
    text = str(value)
    digits = ''.join(ch for ch in text if ch.isdigit())
    return digits.zfill(4)[:4]


def _hhmm_to_minutes(value: str) -> int:
    hhmm = _to_time_str(value)
    return int(hhmm[:2]) * 60 + int(hhmm[2:4])


def _next_trading_day(date_value):
    date_ts = pd.Timestamp(date_value)
    next_day = advanceDateByCalendar('china.sse', date_ts, '+1b')
    return pd.Timestamp(next_day).normalize()


class PositionBacktester(object):

    def __init__(self,
                 market_data: pd.DataFrame,
                 contract_multiplier: float,
                 slippage: float = 0.2,
                 initial_capital: float = 5e7,
                 base_position: int = 10,
                 night_session_start: str = '2100',
                 session_anchor_start: str = '2059'):

        self.market_data = market_data.copy()
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.contract_multiplier = contract_multiplier
        self.base_position = base_position  # 对锁底仓手数

        self.trade_records = None
        self.daily_stats = None

        self._prepare_data(night_session_start=night_session_start,
                           session_anchor_start=session_anchor_start)

    def _prepare_data(self, night_session_start, session_anchor_start):
        self.market_data['date'] = pd.to_datetime(
            self.market_data['trade_time']).dt.normalize()
        self.market_data['min_time'] = self.market_data[
            'trade_time'].dt.strftime('%H%M').astype(str)
        night_start_minutes = _hhmm_to_minutes(night_session_start)
        anchor_start_minutes = _hhmm_to_minutes(session_anchor_start)
        minute_values = self.market_data['min_time'].map(_hhmm_to_minutes)
        self.market_data['session_date'] = self.market_data['date']
        night_mask = minute_values >= night_start_minutes

        if night_mask.any():
            unique_night_dates = self.market_data.loc[
                night_mask, 'date'].drop_duplicates()
            next_day_map = {
                date_value: _next_trading_day(date_value)
                for date_value in unique_night_dates
            }
            self.market_data.loc[night_mask,
                                 'session_date'] = self.market_data.loc[
                                     night_mask, 'date'].map(next_day_map)

        self.market_data['session_sort_key'] = np.where(
            minute_values >= anchor_start_minutes,
            minute_values,
            minute_values + 24 * 60,
        )

        self.market_data = self.market_data.sort_values(
            ['session_date', 'code',
             'session_sort_key']).reset_index(drop=True)

        self.time_index = {}
        for (date,
             code), group in self.market_data.groupby(['session_date',
                                                       'code']):
            group = group.sort_values('session_sort_key')
            times = group['min_time'].drop_duplicates().tolist()
            self.time_index[(date, code)] = times

        self.price_lookup = {}
        for row in self.market_data.itertuples():
            key = (row.session_date, row.code, row.min_time)
            self.price_lookup[key] = row.vwap

        # 获取每个回测交易日的交易时间范围
        self.daily_times = {}
        for (date,
             code), group in self.market_data.groupby(['session_date',
                                                       'code']):
            group = group.sort_values('session_sort_key')
            times = group['min_time'].drop_duplicates().tolist()
            if (date, code) not in self.daily_times:
                self.daily_times[(date, code)] = (times[0], times[-1])

        # 获取所有回测交易日列表
        self.trading_dates = pd.to_datetime(
            sorted(self.market_data['session_date'].unique()))

    def last_price(self, date: int, code: str) -> Optional[float]:
        """获取当日最后一个分钟的 TWAP 作为收盘价"""
        if (date, code) not in self.daily_times:
            return None
        _, last_time = self.daily_times[(date, code)]
        return self.price_lookup.get((date, code, last_time))

    def first_price(self, date: int, code: str) -> Optional[float]:
        """获取当日第一个分钟的 TWAP 作为开盘价"""
        if (date, code) not in self.daily_times:
            return None
        first_time, _ = self.daily_times[(date, code)]
        return self.price_lookup.get((date, code, first_time))

    def next_price(self, date: str, code: str,
                   min_time: str) -> Optional[float]:
        """
        获取下一个分钟的 TWAP 价格（成交价格）
        如果当前是当日最后一个时间，返回 None（无法成交）
        """
        if (date, code) not in self.time_index:
            return None

        times = self.time_index[(date, code)]
        try:
            current_idx = times.index(min_time)
            if current_idx >= len(times) - 1:
                # 当前是当日最后一个时间，无法在下一个分钟成交
                return None
            next_time = times[current_idx + 1]
            return self.price_lookup.get((date, code, next_time))
        except ValueError:
            return None

    def apply_slippage(self, price: float, direction: int) -> float:
        """
        应用滑点到成交价格
        
        - 买入（direction=1）: 成交价 = 理论价 + 滑点
        - 卖出（direction=-1）: 成交价 = 理论价 - 滑点
        """
        if direction == 1:  # 买入
            return price + self.slippage
        else:  # 卖出
            return price - self.slippage

    def calculate_pnl(self, direction: int, exec_price: float,
                      close_price: float, numbers: int,
                      multiplier: float) -> float:
        if direction == 1:  # 多头信号，平空仓（买入平仓）
            pnl = (close_price - exec_price) * numbers * multiplier
        else:  # direction == -1, 空头信号，平多仓（卖出平仓）
            pnl = (exec_price - close_price) * numbers * multiplier
        return pnl

    def run(self, position_df: pd.DataFrame,
            code: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        position_df = position_df.copy()

        # 交易记录
        all_trade_records = []

        # 日度统计
        all_daily_stats = []

        # 累计 PnL
        cumulative_pnl = 0.0

        for date in self.trading_dates[:20]:
            times = self.time_index[(date, code)]
            if len(times) < 120:
                continue
            
            first_time = times[0]
            last_time = times[-1]

            open_price = self.first_price(date, code)
            close_price = self.last_price(date, code)

            if open_price is None or close_price is None:
                continue

            # === 1. 开盘建立对锁底仓 ===
            # 在开盘时建立 base_position 手对锁底仓
            # 多头底仓：以开盘价买入 base_position 手
            # 空头底仓：以开盘价卖出 base_position 手
            # 对锁仓本身不产生 PnL，只是作为底仓
            base_position_record = {
                'date': date,
                'min_time': first_time,
                'code': code,
                'direction': 0,  # 0 表示建立对锁底仓
                'numbers': self.base_position,
                'theoretical_price': open_price,
                'exec_price': open_price,
                'close_price': close_price,
                'multiplier': self.contract_multiplier,
                'pnl': 0.0,  # 对锁仓不产生 PnL
                'signal_type': 'base_position'
            }
            all_trade_records.append(base_position_record)

            # === 2. 处理交易信号 ===
            # 使用传入的信号
            day_signals = position_df[position_df['date'] == date]
            # 当日交易统计
            daily_long_trades = 0  # 多头交易手数
            daily_short_trades = 0  # 空头交易手数
            daily_pnl = 0.0
            # 处理每个信号
            if len(day_signals) > 0:
                for signal in day_signals.itertuples():
                    min_time = signal.min_time
                    direction = signal.direction
                    numbers = signal.numbers

                    theoretical_price = self.next_price(date, code, min_time)
                    if theoretical_price is None:
                        # 无法成交（当日最后一个信号）
                        continue

                    # 应用滑点
                    exec_price = self.apply_slippage(theoretical_price,
                                                     direction)

                    pnl = self.calculate_pnl(direction, exec_price,
                                             close_price, numbers,
                                             self.contract_multiplier)
                    # 记录交易
                    trade_record = {
                        'date': date,
                        'min_time': min_time,
                        'code': code,
                        'direction': direction,
                        'numbers': numbers,
                        'theoretical_price': theoretical_price,
                        'exec_price': exec_price,
                        'close_price': close_price,
                        'multiplier': self.contract_multiplier,
                        'pnl': pnl,
                        'signal_type': 'regular'
                    }
                    all_trade_records.append(trade_record)
                    # 统计
                    if direction == 1:
                        daily_long_trades += numbers
                    else:
                        daily_short_trades += numbers
                    daily_pnl += pnl

            net_exposure = daily_long_trades - daily_short_trades
            # 如果有敞口，需要平仓
            close_pnl = 0.0
            if net_exposure != 0:
                # 净敞口 > 0: 多头过多，需要卖出平仓
                # 净敞口 < 0: 空头过多，需要买入平仓
                close_direction = -1 if net_exposure > 0 else 1
                close_numbers = abs(net_exposure)
                close_pnl = self.calculate_pnl(close_direction, close_price,
                                               close_price, close_numbers,
                                               self.contract_multiplier)
                close_record = {
                    'date': date,
                    'min_time': last_time,
                    'code': code,
                    'direction': close_direction,
                    'numbers': close_numbers,
                    'theoretical_price': close_price,
                    'exec_price': close_price,
                    'close_price': close_price,
                    'multiplier': self.contract_multiplier,
                    'pnl': close_pnl,
                    'signal_type': 'close_position'
                }
                all_trade_records.append(close_record)
                daily_pnl += close_pnl

            # 更新累计 PnL
            cumulative_pnl += daily_pnl
            # 计算净值
            nav = (self.initial_capital +
                   cumulative_pnl) / self.initial_capital
            # 日度统计
            all_daily_stats.append({
                'date': date,
                'code': code,
                'long_trades': daily_long_trades,
                'short_trades': daily_short_trades,
                'net_exposure': net_exposure,
                'daily_pnl': daily_pnl,
                'cumulative_pnl': cumulative_pnl,
                'nav': nav,
                'open_price': open_price,
                'close_price': close_price
            })

        # 转换为 DataFrame
        self.trade_records = pd.DataFrame(all_trade_records)
        self.daily_stats = pd.DataFrame(all_daily_stats)
        return self.trade_records, self.daily_stats

    def metrics(self) -> dict:
        if self.daily_stats is None or len(self.daily_stats) == 0:
            return {}

        df = self.daily_stats

        total_days = len(df)
        win_days = (df['daily_pnl'] > 0).sum()
        lose_days = (df['daily_pnl'] < 0).sum()
        win_rate = win_days / total_days * 100

        total_pnl = df['cumulative_pnl'].iloc[-1]
        final_nav = df['nav'].iloc[-1]

        # 计算最大回撤
        df['cum_nav'] = df['nav'].cummax()
        df['drawdown'] = (df['cum_nav'] - df['nav']) / df['cum_nav'] * 100
        max_drawdown = df['drawdown'].max()

        # 计算夏普比率（假设无风险利率为 0）
        daily_returns = df['daily_pnl'] / self.initial_capital
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std(
        ) if daily_returns.std() > 0 else 0

        return {
            'total_days': total_days,
            'win_days': win_days,
            'lose_days': lose_days,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_nav': final_nav,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
