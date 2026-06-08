import pandas as pd
import datetime, pdb

# ================= 配置区 =================
KLINE_MINUTE_WINDOW = 5  # 设置为你想要的分钟数 (1, 5, 15, 30 等)
# ==========================================

class CacheBar(object):
    def __init__(self, symbol):
        self.symbol = symbol
        self.bar = None
        # 【修改 1】：不再只记录分钟，而是记录完整的“当前周期的起始时间”
        self.current_window_start = None  
        self.prev_total_volume = 0.0
        self.prev_total_turnover = 0.0

class BarData(object):
    def __init__(self):
        self.vt_symbol = ""
        self.symbol = ""
        self.exchange = ""
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.date = ""
        self.time = ""
        self.datetime = None
        self.volume = 0.0
        self.value = 0.0
        self.open_interest = 0.0

class TickData(object):
    def __init__(self, row):
        self.symbol = row['InstrumentID']
        self.vt_symbol = f"{self.symbol}.SHFE" 
        self.exchange = "SHFE" 
        date_str = str(row['TradingDay'])
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}" 
        self.datetime = datetime.datetime.strptime(f"{date_str} {row['UpdateTime']}", '%Y-%m-%d %H:%M:%S')
        self.last_price = float(row['LastPrice'])
        self.volume = float(row['Volume'])
        self.turnover = float(row['Turnover'])
        self.open_interest = float(row['OpenInterest'])


bars = {}
result_bars = []

# 【核心功能】：计算任意时间所属的 K 线周期起点
def get_window_start_time(tick_time, window_minutes):
    """
    例如: tick_time = 09:32:15, window_minutes = 5
    返回: 09:30:00 (datetime 对象)
    """
    # 计算当前分钟数是 5 的多少倍，并向下取整
    minute_floor = (tick_time.minute // window_minutes) * window_minutes
    # 返回替换了分钟和秒的时间
    return tick_time.replace(minute=minute_floor, second=0, microsecond=0)


def process_tick(tick):
    global bars, result_bars
    
    if tick.symbol not in bars:
        bars[tick.symbol] = CacheBar(tick.symbol)
        
    cache_bar = bars[tick.symbol]
    tick_time = tick.datetime
    
    # 【修改 2】：计算当前 Tick 属于哪一个 K 线窗口
    tick_window_start = get_window_start_time(tick_time, KLINE_MINUTE_WINDOW)
    
    if not hasattr(cache_bar, 'prev_total_volume') or cache_bar.prev_total_volume == 0.0:
        cache_bar.prev_total_volume = tick.volume
        cache_bar.prev_total_turnover = tick.turnover

    # ================= 周期跳变逻辑 =================
    # 判断当前的窗口起始时间，是否和缓存的不一致
    if (cache_bar.current_window_start is not None) and (tick_window_start != cache_bar.current_window_start):
        if cache_bar.bar:
            data = cache_bar.bar.__dict__.copy()

            current_total_vol = data['volume']
            current_total_turnover = data['value']

            data['volume'] = max(0.0, current_total_vol - cache_bar.prev_total_volume)
            data['value'] = max(0.0, current_total_turnover - cache_bar.prev_total_turnover)

            multiplier = 15 
            if data['volume'] > 0:
                data['vwap'] = data['value'] / data['volume'] / multiplier
            else:
                data['vwap'] = data['close'] 

            print(f"✅ 生成 {KLINE_MINUTE_WINDOW}分钟 K线: {data['datetime']} | "
                  f"O:{data['open']} H:{data['high']} L:{data['low']} C:{data['close']} | "
                  f"Vol:{data['volume']} Val:{data['value']:.2f}")
            
            result_bars.append(data)

            cache_bar.prev_total_volume = current_total_vol
            cache_bar.prev_total_turnover = current_total_turnover

        # ---------------- 创建新的一根 N 分钟 Bar ----------------
        bar = BarData()
        bar.vt_symbol = tick.vt_symbol
        bar.symbol = tick.symbol
        bar.exchange = tick.exchange
        
        bar.open = tick.last_price
        bar.high = tick.last_price
        bar.low = tick.last_price
        bar.close = tick.last_price
        
        # 【修改 3】：K 线的标记时间，统一使用该周期的起始时间！
        # 比如这根 K 线装的是 09:30 到 09:35 的数据，它的名字统一叫 '09:30:00'
        bar.date = tick_window_start.strftime('%Y-%m-%d')
        bar.time = tick_window_start.strftime('%H:%M:%S')
        bar.datetime = tick_window_start.strftime('%Y-%m-%d %H:%M:%S')

        bar.volume = tick.volume
        bar.value = tick.turnover
        bar.open_interest = tick.open_interest

        cache_bar.bar = bar
        # 更新当前的周期标记
        cache_bar.current_window_start = tick_window_start

    # ================= 同一周期内部 =================
    else:
        if cache_bar.bar is None:
            bar = BarData()
            bar.vt_symbol = tick.vt_symbol
            bar.symbol = tick.symbol
            bar.exchange = tick.exchange
            
            bar.open = tick.last_price
            bar.high = tick.last_price
            bar.low = tick.last_price
            bar.close = tick.last_price
            
            bar.date = tick_window_start.strftime('%Y-%m-%d')
            bar.time = tick_window_start.strftime('%H:%M:%S')
            bar.datetime = tick_window_start.strftime('%Y-%m-%d %H:%M:%S')

            bar.volume = tick.volume
            bar.value = tick.turnover
            bar.open_interest = tick.open_interest
            
            cache_bar.bar = bar
            cache_bar.current_window_start = tick_window_start
        else:
            bar = cache_bar.bar
            bar.high = max(bar.high, tick.last_price)
            bar.low = min(bar.low, tick.last_price)
            bar.close = tick.last_price

            bar.volume = tick.volume
            bar.value = tick.turnover
            bar.open_interest = tick.open_interest

    bars[tick.symbol] = cache_bar


# 模拟运行
print(f"开始处理，目标频率: {KLINE_MINUTE_WINDOW}分钟...")
dt = pd.read_csv('ag20260424.csv')
for index, row in dt.iterrows():
    tick_obj = TickData(row)
    process_tick(tick_obj)
    
pdb.set_trace()
if result_bars:
    final_df = pd.DataFrame(result_bars)
    print("\n最终合成的 Bar 数据：")
    print(final_df[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap']].head(10))