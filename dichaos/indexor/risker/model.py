from pydantic import BaseModel, Field


class Risker(BaseModel):
    date: str
    remaining_position_limit: float
    current_price: float
    portfolio_value: float
    current_position: float
    position_limit: float
    remaining_limit: float
    available_cash: float


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
