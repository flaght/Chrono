### 数据集放入进去 所有的指标算一遍，回测的话 返回批量周期因子，实盘的返回最新因子
from .calculate import *
class Builder:
    def calcuate_sma(self, data: pd.DataFrame):
        """
        Calculate the Simple Moving Average (SMA) for a given period.
        """
        sma5 = calcuate_sma(data, period=5)
        sma10 = calcuate_sma(data, period=10)
        sma20 = calcuate_sma(data, period=20)
        return sma5, sma10, sma20

    def calculate_ema(self, data: pd.DataFrame):
        """
        Calculate the Exponential Moving Average (EMA) for a given period.
        """
        ema12 = calculate_ema(data, period=12)
        ema26 = calculate_ema(data, period=26)
        return ema12, ema26

    def calculate_rsi(self, data: pd.DataFrame):
        """
        Calculate the Relative Strength Index (RSI) for a given period.
        """
        return calculate_rsi(data, period=14)

    def calculate_macd(self, data: pd.DataFrame):
        """
        Calculate the Moving Average Convergence Divergence (MACD) for a given period.
        """
        return calculate_macd(data,
                              fast_period=12,
                              slow_period=26,
                              signal_period=9)

    def calculate_bollinger_bands(self, data: pd.DataFrame):
        """
        Calculate the Bollinger Bands for a given period.
        """
        return calculate_bollinger_bands(data, period=20, std_dev=2)

    def calculate_atr(self, data: pd.DataFrame):
        """
        Calculate the Average True Range (ATR) for a given period.
        """
        return calculate_atr(data, period=14)

    def calculate_vwap(self, data: pd.DataFrame):
        """
        Calculate the Volume Weighted Average Price (VWAP) for a given period.
        """
        return calculate_vwap(data)

    def calculate_adx(self, data: pd.DataFrame):
        """
        Calculate the Average Directional Index (ADX) for a given period.
        """
        return calculate_adx(data, period=14)

    def calculate_obv(self, data: pd.DataFrame):
        """
        Calculate the On-Balance Volume (OBV) for a given period.
        """
        return calculate_obv(data)

    def calcuate_point(self, data: pd.DataFrame):
        """
        Calculate the point for a given period.
        """
        point = calcuate_point(data)
        return point