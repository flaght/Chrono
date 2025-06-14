try:
    import talib
    TALIB_AVAILABLE = True
except ImportError as e:
    TALIB_AVAILABLE = False
    print(f"TA-Lib is not available: {e}")
import numpy as np
import pandas as pd
import pdb


def calcuate_point(df):
    pp = (df['high'] + df['low'] + df['close']) / 3
    r1 = 2 * pp - df['low']
    s1 = 2 * pp - df['high']
    r2 = pp + (df['high'] - df['low'])
    s2 = pp - (df['high'] - df['low'])
    r3 = df['high'] + 2 * (pp - df['low'])
    s3 = df['low'] - 2 * (df['high'] - pp)
    return pp.dropna().round(4), r1.dropna().round(4), s1.dropna().round(4), r2.dropna().round(4), s2.dropna(
    ).round(4), r3.dropna().round(4), s3.dropna().round(4)


def calcuate_sma(df, period):
    """
    Calculate the Simple Moving Average (SMA) for a given period.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        period (int): The number of periods to calculate the SMA.

    Returns:
        pd.Series: Series containing the calculated SMA.
    """
    if TALIB_AVAILABLE:
        return talib.SMA(df['close'], timeperiod=period)
    else:
        return df['close'].rolling(window=period,
                                   min_periods=1).mean().dropna().round(4)


def calculate_ema(df, period):
    """
    Calculate the Exponential Moving Average (EMA) for a given period.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        period (int): The number of periods to calculate the EMA.

    Returns:
        pd.Series: Series containing the calculated EMA.
    """
    if TALIB_AVAILABLE:
        return talib.EMA(df['close'], timeperiod=period)
    else:
        return df['close'].ewm(span=period, adjust=False, min_periods=1).mean().dropna().round(4)


def calculate_rsi(df, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given period.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        period (int): The number of periods to calculate the RSI.

    Returns:
        pd.Series: Series containing the calculated RSI.
    """
    if TALIB_AVAILABLE:
        return talib.RSI(df['close'], timeperiod=period)
    else:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        # 计算RS和RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)  # 避免除以零
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).dropna().round(4)  # 填充NaN值为50


def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate the MACD (Moving Average Convergence Divergence) indicator.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        fast_period (int): The period for the fast EMA.
        slow_period (int): The period for the slow EMA.
        signal_period (int): The period for the signal line.

    Returns:
        pd.Series: Series containing the MACD values.
    """
    if TALIB_AVAILABLE:
        macd, signal, hist = talib.MACD(df['close'],
                                        fastperiod=fast_period,
                                        slowperiod=slow_period,
                                        signalperiod=signal_period)
        return macd
    else:
        ema_fast = df['close'].ewm(span=fast_period,
                                   adjust=False,
                                   min_periods=1).mean()
        ema_slow = df['close'].ewm(span=slow_period,
                                   adjust=False,
                                   min_periods=1).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False,
                          min_periods=1).mean()
        hist = macd - signal
        return pd.DataFrame({'macd': macd, 'signal': signal, 'hist': hist}).dropna().round(4)


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculate the Bollinger Bands for a given period.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        period (int): The number of periods to calculate the Bollinger Bands.
        std_dev (int): The number of standard deviations to use.

    Returns:
        pd.DataFrame: DataFrame containing the upper and lower bands.
    """
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(df['close'],
                                            timeperiod=period,
                                            nbdevup=std_dev,
                                            nbdevdn=std_dev,
                                            matype=0)
    else:
        # 手动计算布林带
        middle = df['close'].rolling(window=period, min_periods=1).mean()
        std = df['close'].rolling(window=period, min_periods=1).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

    return pd.DataFrame({'middle': middle, 'upper': upper, 'lower': lower}).dropna().round(4)


def calculate_kdj(df, period=14):
    """
    Calculate the KDJ indicator.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        period (int): The number of periods to calculate the KDJ.

    Returns:
        pd.DataFrame: DataFrame containing K, D, and J values.
    """
    if TALIB_AVAILABLE:
        kdj = talib.STOCHF(df['high'],
                           df['low'],
                           df['close'],
                           fastk_period=period,
                           fastd_period=3,
                           fastd_matype=0)
        return pd.DataFrame({'k': kdj['fastk'], 'd': kdj['fastd']})
    else:
        # 手动计算KDJ
        low_min = df['low'].rolling(window=period, min_periods=1).min()
        high_max = df['high'].rolling(window=period, min_periods=1).max()
        rsv = 100 * ((df['close'] - low_min) / (high_max - low_min))
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        return pd.DataFrame({'k': k, 'd': d, 'j': j}).dropna().round(4)


def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR) for a given period.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        period (int): The number of periods to calculate the ATR.

    Returns:
        pd.Series: Series containing the calculated ATR.
    """

    # 检查输入数据的有效性
    if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        missing = [
            col for col in ['high', 'low', 'close'] if col not in df.columns
        ]
        return pd.Series(index=df.index)  # 返回空Series

    # 使用TA-Lib (如可用)或手动计算
    if TALIB_AVAILABLE:
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
    else:
        # 计算三种真实范围
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))

        # 取三种TR中的最大值
        true_range = pd.concat([high_low, high_close_prev, low_close_prev],
                               axis=1).max(axis=1)

        # 使用滚动平均计算ATR，min_periods=1确保尽可能计算
        atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr.dropna().round(4)


def calculate_vwap(df):
    """
    Calculate the Volume Weighted Average Price (VWAP).

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        pd.Series: Series containing the calculated VWAP.
    """
    if 'volume' not in df.columns:
        return pd.Series(index=df.index)  # 返回空Series

    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap.dropna().round(4)


def calculate_adx(df, period=14):
    """
    Calculate the Average Directional Index (ADX).

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        pd.Series: Series containing the calculated ADX.
    """
    if TALIB_AVAILABLE:
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
    else:
        # 手动计算ADX
        high = df['high']
        low = df['low']
        close = df['close']

        # 计算True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算+DI和-DI
        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr14 = true_range.rolling(window=period, min_periods=1).mean()
        plus_dm14 = plus_dm.rolling(window=period, min_periods=1).mean()
        minus_dm14 = minus_dm.rolling(window=period, min_periods=1).mean()

        plus_di = 100 * (plus_dm14 / tr14)
        minus_di = -100 * (minus_dm14 / tr14)

        # 计算ADX
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period, min_periods=1).mean()

    return adx.dropna().round(4)

def calculate_obv(df):
    """
    Calculate the On-Balance Volume (OBV).

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        pd.Series: Series containing the calculated OBV.
    """
    if 'volume' not in df.columns:
        return pd.Series(index=df.index)  # 返回空Series

    obv = (df['volume'] * np.sign(df['close'].diff())).cumsum()
    return obv.dropna().round(4)
