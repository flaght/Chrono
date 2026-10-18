### 模拟回测示例代码

import pdb
from typing import Tuple, Optional
import pandas as pd
import numpy as np

def select_main_contract(df):
    """筛选每日主力合约（Code 最大值）"""
    return df.groupby('date', group_keys=False).apply(
        lambda x: x[x['Code'] == x['Code'].max()]
    ).reset_index(drop=True)
    
def generate_position_signals(price_data: pd.DataFrame, interval: int = 15) -> pd.DataFrame:
    """
    生成模拟仓位信号数据
    
    参数:
    - price_data: pd.DataFrame, 行情数据（主力合约数据）
    - interval: int, 信号频率（分钟），默认 15 分钟
    
    返回:
    - position_df: pd.DataFrame, 仓位信号
      columns: ['date', 'minTime', 'Code', 'direction', 'numbers']
    
    信号规则:
    - 从 09:46 开始（586 分钟），每隔 interval 分钟生成一个信号
    - 交替生成多头和空头信号
    - 每次 1 手
    """
    time_index = {}
    for (date, code), group in price_data.groupby(['date', 'Code']):
        times = sorted(group['minTime'].unique())
        time_index[(date, code)] = times
        
    all_signals = []
    
    for (date, code), times in time_index.items():
        signals = []
        
        for time in times:
            # 转换为分钟数
            hour = int(time[:2])
            minute = int(time[2:4])
            total_minutes = hour * 60 + minute
            
            # 从 09:46 开始（586 分钟）
            if total_minutes < 586:
                continue
            
            if (total_minutes - 586) % interval == 0:
                direction = 1 if (len(signals) % 2 == 0) else -1
                signals.append({
                    'date': date,
                    'minTime': time,
                    'Code': code,
                    'direction': direction,
                    'numbers': 1
                })
                
        all_signals.extend(signals)
    return pd.DataFrame(all_signals)
        
    

class PositionBacktester:
    IM_MULTIPLIER = 200
    def __init__(self, 
                 price_data: pd.DataFrame,
                 slippage: float = 0.2,
                 initial_capital: float = 5e7,
                 base_position: int = 10):
        
        self.price_data = price_data.copy()
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.base_position = base_position  # 对锁底仓手数
        
        # 合约乘数
        self.contract_multiplier = self.IM_MULTIPLIER
        
        # 预处理数据：构建查找索引
        self._prepare_data()
        
        # 回测结果
        self.trade_records = None
        self.daily_stats = None
        
    def _prepare_data(self):
        # 按 date, Code, minTime 排序
        self.price_data = self.price_data.sort_values(['date', 'Code', 'minTime']).reset_index(drop=True)
        
        # 构建 (date, Code) -> minTime 列表的映射
        self.time_index = {}
        for (date, code), group in self.price_data.groupby(['date', 'Code']):
            times = sorted(group['minTime'].unique())
            self.time_index[(date, code)] = times
            
        # 构建快速查找表：(date, Code, minTime) -> twap
        self.price_lookup = {}
        for _, row in self.price_data.iterrows():
            key = (row['date'], row['Code'], row['minTime'])
            self.price_lookup[key] = row['twap']
            
        # 获取每个交易日的交易时间范围
        self.daily_times = {}
        for (date, code), group in self.price_data.groupby(['date', 'Code']):
            times = sorted(group['minTime'].unique())
            if (date, code) not in self.daily_times:
                self.daily_times[(date, code)] = (times[0], times[-1])
        
        # 获取所有交易日列表
        self.trading_dates = sorted(self.price_data['date'].unique())
    
    
    def _get_main_contract(self, date: int) -> Optional[str]:
        """获取指定日期的主力合约代码"""
        day_data = self.price_data[self.price_data['date'] == date]
        if len(day_data) == 0:
            return None
        return day_data['Code'].max()
    
    def _get_close_twap(self, date: int, code: str) -> Optional[float]:
        """获取当日最后一个分钟的 TWAP 作为收盘价"""
        if (date, code) not in self.daily_times:
            return None
        
        _, last_time = self.daily_times[(date, code)]
        return self.price_lookup.get((date, code, last_time))
    
    def _get_open_twap(self, date: int, code: str) -> Optional[float]:
        """获取当日第一个分钟的 TWAP 作为开盘价"""
        if (date, code) not in self.daily_times:
            return None
        first_time, _ = self.daily_times[(date, code)]
        
        return self.price_lookup.get((date, code, first_time))
    
    def _get_next_twap(self, date: str, code: str, min_time: str) -> Optional[float]:
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
    
    def _apply_slippage(self, price: float, direction: int) -> float:
        """
        应用滑点到成交价格
        
        - 买入（direction=1）: 成交价 = 理论价 + 滑点
        - 卖出（direction=-1）: 成交价 = 理论价 - 滑点
        """
        if direction == 1:  # 买入
            return price + self.slippage
        else:  # 卖出
            return price - self.slippage
    
    
    def _calculate_pnl(self, 
                       direction: int, 
                       exec_price: float, 
                       close_price: float, 
                       numbers: int, 
                       multiplier: float) -> float:
        if direction == 1:  # 多头信号，平空仓（买入平仓）
            pnl = (close_price - exec_price) * numbers * multiplier
        else:  # direction == -1, 空头信号，平多仓（卖出平仓）
            pnl = (exec_price - close_price) * numbers * multiplier
        return pnl
        
        
    def run_backtest(self, position_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        position_df = position_df.copy()
        
         # 交易记录
        all_trade_records = []
        
        # 日度统计
        all_daily_stats = []
        
        # 累计 PnL
        cumulative_pnl = 0.0
        
        pdb.set_trace()
        for date in self.trading_dates[:2]:
            # 获取当日主力合约
            code = self._get_main_contract(date)
            if code is None:
                continue
            
            if (date, code) not in self.time_index:
                continue
            
            times = self.time_index[(date, code)]
            first_time = times[0]
            last_time = times[-1]
            
            # 
            open_price = self._get_open_twap(date, code)
            close_price = self._get_close_twap(date, code)
            if open_price is None or close_price is None:
                continue
            
            # === 1. 开盘建立对锁底仓 ===
            # 在开盘时建立 base_position 手对锁底仓
            # 多头底仓：以开盘价买入 base_position 手
            # 空头底仓：以开盘价卖出 base_position 手
            # 对锁仓本身不产生 PnL，只是作为底仓
            
            base_position_record = {
                'date': date,
                'minTime': first_time,
                'Code': code,
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
                for _, signal in day_signals.iterrows():
                    min_time = signal['minTime']
                    direction = signal['direction']
                    numbers = signal['numbers']
                    
                    # 获取下一个分钟的 TWAP 作为理论成交价
                    theoretical_price = self._get_next_twap(date, code, min_time)
                    
                    if theoretical_price is None:
                        # 无法成交（当日最后一个信号）
                        continue
                    
                    # 应用滑点
                    exec_price = self._apply_slippage(theoretical_price, direction)
                    
                    pnl = self._calculate_pnl(direction, exec_price, close_price, numbers, self.contract_multiplier)
                    
                    # 记录交易
                    trade_record = {
                        'date': date,
                        'minTime': min_time,
                        'Code': code,
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
                
                # 以收盘价平仓
                close_pnl = self._calculate_pnl(close_direction, close_price, close_price, close_numbers, self.contract_multiplier)
               
                close_record = {
                'date': date,
                'minTime': last_time,
                'Code': code,
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
            nav = (self.initial_capital + cumulative_pnl) / self.initial_capital
            
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
        
        pdb.set_trace()
        # 转换为 DataFrame
        self.trade_records = pd.DataFrame(all_trade_records)
        self.daily_stats = pd.DataFrame(all_daily_stats)
        
        return self.trade_records, self.daily_stats
            
        
    def calculate_performance_metrics(self) -> dict:
        """计算绩效指标"""
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
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
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
        
        
        
        
price_data = pd.read_parquet('/home/dev1/future_min1/20200102_20250313_bar.parquet')[['twap']].reset_index()
# print("行情数据形状:", price_data.shape)
# print("日期范围:", price_data['date'].min(), "-", price_data['date'].max())

pdb.set_trace()
im_data = price_data[price_data['Code'].str.startswith('IM')].copy()
# print("\nIM 合约数据形状:", im_data.shape)
# print("IM 合约日期范围:", im_data['date'].min(), "-", im_data['date'].max())
# print("IM 合约数量:", im_data['Code'].nunique())



main_contract_data = select_main_contract(im_data)
# print("\n主力合约数据形状:", main_contract_data.shape)
# print("交易日数量:", main_contract_data['date'].nunique())
# print("\n主力合约示例:")
main_contract_data.head(10)

main_contracts_by_date = main_contract_data.groupby('date')['Code'].first()
# print("\n主力合约换月情况:")
# print(main_contracts_by_date.head(30))
# pdb.set_trace()
contract_counts = main_contracts_by_date.value_counts()
# print("\n主力合约交易日统计:")
# print(contract_counts.head())

pdb.set_trace()
signals_15min = generate_position_signals(main_contract_data, interval=15)
print(f"\n15 分钟信号数量：{len(signals_15min)}")
print("信号示例:")
print(signals_15min.head(10))

# 测试第一个交易日的信号
test_date = main_contract_data['date'].unique()[0]
test_signals = signals_15min[signals_15min['date'] == test_date]
print(f"\n日期 {test_date} 的信号:")
print(f"信号数量：{len(test_signals)}")
print(test_signals)

pdb.set_trace()

backtester = PositionBacktester(
    price_data=main_contract_data,
    slippage=0.2,           # 0.2 个指数点滑点
    initial_capital=5e7,    # 5000 万初始资金
    base_position=10        # 10 手对锁底仓
)

print("回测引擎初始化完成")
print(f"IM 合约乘数：{backtester.contract_multiplier}")
print(f"初始资金：{backtester.initial_capital:,.0f}")
print(f"对锁底仓：{backtester.base_position} 手")


trade_records2, daily_stats2 = backtester.run_backtest(signals_15min)
metrics2 = backtester.calculate_performance_metrics()
pdb.set_trace()
print('-->')
