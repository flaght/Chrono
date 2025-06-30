from pydantic import BaseModel, Field
from typing_extensions import Literal
import pdb


class Indicator(BaseModel):
    name: str  ## 指标名称
    id: str  ## 显示
    values: float  ## 指标值


class IndicatorList(BaseModel):
    sma5: Indicator = Field(default=None, nullable=True)
    sma10: Indicator = Field(default=None, nullable=True)
    sma20: Indicator = Field(default=None, nullable=True)
    ema12: Indicator = Field(default=None, nullable=True)
    ema26: Indicator = Field(default=None, nullable=True)
    rsi: Indicator = Field(default=None, nullable=True)
    macd: Indicator = Field(default=None, nullable=True)
    atr: Indicator = Field(default=None, nullable=True)
    vwap: Indicator = Field(default=None, nullable=True)
    adx: Indicator = Field(default=None, nullable=True)
    obv: Indicator = Field(default=None, nullable=True)
    pp: Indicator = Field(default=None, nullable=True)
    r1: Indicator = Field(default=None, nullable=True)
    s1: Indicator = Field(default=None, nullable=True)
    r2: Indicator = Field(default=None, nullable=True)
    s2: Indicator = Field(default=None, nullable=True)
    r3: Indicator = Field(default=None, nullable=True)
    s3: Indicator = Field(default=None, nullable=True)
    date: str  ## 日期

    def update(self, indicator: Indicator):
        """
        Update the indicator list with a new indicator.
        """
        if hasattr(self, indicator.id):
            setattr(self, indicator.id, indicator)
        else:
            raise ValueError(
                f"Indicator {indicator.id} not found in the list.")

    def set_indicator(self, **kwargs):
        self.update(
            Indicator(id='sma5',
                      name='SMA5',
                      values=kwargs.get('sma5').values[0]))
        self.update(
            Indicator(id='sma10',
                      name='SMA10',
                      values=kwargs.get('sma10').values[0]))

        self.update(
            Indicator(id='sma20',
                      name='SMA20',
                      values=kwargs.get('sma20').values[0]))

        self.update(
            Indicator(id='ema12',
                      name='EMA12',
                      values=kwargs.get('ema12').values[0]))

        self.update(
            Indicator(id='ema26',
                      name='EMA26',
                      values=kwargs.get('ema26').values[0]))

        self.update(
            Indicator(id='rsi',
                      name='RSI14',
                      values=kwargs.get('rsi').values[0]))

        self.update(
            Indicator(id='macd',
                      name='MACD',
                      values=kwargs.get('macd').values[0][0]))

        self.update(
            Indicator(id='atr',
                      name='ATR14',
                      values=kwargs.get('atr').values[0]))

        self.update(
            Indicator(id='vwap',
                      name='VWAP',
                      values=kwargs.get('vwap').values[0]))

        self.update(
            Indicator(id='adx',
                      name='ADX14',
                      values=kwargs.get('adx').values[0]))

        self.update(
            Indicator(id='obv', name='OBV',
                      values=kwargs.get('obv').values[0]))

        self.update(
            Indicator(id='pp', name='PP', values=kwargs.get('pp').values[0]))

        self.update(
            Indicator(id='r1', name='PR1', values=kwargs.get('r1').values[0]))

        self.update(
            Indicator(id='s1', name='PS1', values=kwargs.get('s1').values[0]))

        self.update(
            Indicator(id='r2', name='PR2', values=kwargs.get('r2').values[0]))

        self.update(
            Indicator(id='s2', name='PS2', values=kwargs.get('s2').values[0]))

        self.update(
            Indicator(id='r3', name='PR3', values=kwargs.get('r3').values[0]))

        self.update(
            Indicator(id='s3', name='PS3', values=kwargs.get('s3').values[0]))

    def format(self):
        text = "{0} 技术指标:\n".format(self.date)
        text += "{0}: {1} \n".format(self.sma5.name, self.sma5.values)
        text += "{0}: {1} \n".format(self.sma10.name, self.sma10.values)
        text += "{0}: {1} \n".format(self.sma20.name, self.sma20.values)
        text += "{0}: {1} \n".format(self.ema12.name, self.ema12.values)
        text += "{0}: {1} \n".format(self.ema26.name, self.ema26.values)
        text += "{0}: {1} \n".format(self.rsi.name, self.rsi.values)
        text += "{0}: {1} \n".format(self.macd.name, self.macd.values)
        text += "{0}: {1} \n".format(self.atr.name, self.atr.values)
        text += "{0}: {1} \n".format(self.vwap.name, self.vwap.values)
        text += "{0}: {1} \n".format(self.adx.name, self.adx.values)
        text += "{0}: {1} \n".format(self.obv.name, self.obv.values)

        text += "{0}: Pivot Points:\n".format(self.date)
        text += "{0}: {1} \n".format(self.pp.name, self.pp.values)
        text += "{0}: {1} \n".format(self.r1.name, self.r1.values)
        text += "{0}: {1} \n".format(self.s1.name, self.s1.values)
        text += "{0}: {1} \n".format(self.r2.name, self.r2.values)
        text += "{0}: {1} \n".format(self.s2.name, self.s2.values)
        text += "{0}: {1} \n".format(self.r3.name, self.r3.values)
        text += "{0}: {1} \n".format(self.s3.name, self.s3.values)

        return text


class KLine(BaseModel):
    date: str
    symbol: str
    open: float = Field(default=None, nullable=True)
    close: float = Field(default=None, nullable=True)
    high: float = Field(default=None, nullable=True)
    low: float = Field(default=None, nullable=True)
    volume: float = Field(default=None, nullable=True)

    def format(self):
        text = "{0} K线数据:\n".format(self.date)
        text += "开盘价: {0}\n".format(self.open)
        text += "收盘价: {0}\n".format(self.close)
        text += "最高价: {0}\n".format(self.high)
        text += "最低价: {0}\n".format(self.low)
        text += "成交量: {0}\n".format(self.volume)
        return text


class Memory(BaseModel):
    date: str
    symbol: str
    index: str = Field(default=None, nullable=True)
    indicator: IndicatorList = Field(default=None, nullable=True)
    kline: KLine = Field(default=None, nullable=True)

    def format(self, types: str = "short"):
        desc = {"short": "短期记忆"}
        sidx = {"short": "S"}

        text = "{0}索引ID: {1}{2}\n".format(desc[types], sidx[types], self.index)
        text += self.indicator.format()
        text += self.kline.format()
        return text
