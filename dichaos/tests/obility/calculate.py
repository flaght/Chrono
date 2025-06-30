try:
    import talib
    TALIB_AVAILABLE = True
except ImportError as e:
    TALIB_AVAILABLE = False
    print(f"TA-Lib is not available: {e}")
import numpy as np
import pandas as pd
import pdb



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