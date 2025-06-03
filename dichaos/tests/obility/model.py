from pydantic import BaseModel, Field
from typing_extensions import Literal
import pdb


class Indicator(BaseModel):
    name: str  ## 指标名称
    id: str  ## 显示
    values: float  ## 指标值


class IndicatorList(BaseModel):
    rsi: Indicator = Field(default=None, nullable=True)
    macd: Indicator = Field(default=None, nullable=True)
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
            Indicator(id='rsi',
                      name='RSI14',
                      values=kwargs.get('rsi').values[0]))

        self.update(
            Indicator(id='macd',
                      name='MACD',
                      values=kwargs.get('macd').values[0][0]))


    def format(self):
        text = "{0} 技术指标:\n".format(self.date)
        text += "{0}: {1} \n".format(self.rsi.name, self.rsi.values)
        text += "{0}: {1} \n".format(self.macd.name, self.macd.values)
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


def create_prediction_dom(short_prompt, mid_prompt, long_prompt,
                          reflection_prompt):

    class DomInfo(BaseModel):
        if len(short_prompt) > 0:
            short_memory_index: str = Field(
                ...,
                description="短期记忆的索引ID,格式必须为:'S1,S2,S3',如果没有引用，则不返回 不能随意添加",
                define="S1,S2,S3",
            )
        if len(mid_prompt) > 0:
            mid_memory_index: str = Field(
                ...,
                description="中期记忆的索引ID,格式必须为:'M1,M2,M3',如果没有引用，则不返回 不能随意添加",
                define="M1,M2,M3",
            )
        if len(long_prompt) > 0:
            long_memory_index: str = Field(
                ...,
                description="长期记忆的索引ID,格式必须为:'L1,L2,L3',如果没有引用，则不返回 不能随意添加",
                define="L1,L2,L3",
            )
        if len(reflection_prompt) > 0:
            reflection_memory_index: str = Field(
                ...,
                description="过去反思记忆的索引ID,格式必须为:'R1,R2,R3',如果没有引用，则不返回 不能随意添加",
                define="R1,R2,R3",
            )
        reasoning: str = Field(
            ...,
            description="交易信号的理由, 请提供详细的解释,而且必须返回内容",
            define="string",
        )

        confidence: float = Field(
            ...,
            description="信号置信度",
            define="int",
        )

        signal: Literal["bullish", "bearish", "neutral"] = Field(
            ...,
            description="基于市场数据和分析, 请做出投资决策：买入标的、卖出标的，持有标的",
            define="bullish/bearish/neutral",
        )
        analysis_details: str = Field(
            ...,
            description="技术分析细节, 请提供详细的解释,而且必须返回内容",
            define="string",
        )

        @classmethod
        def dumps(cls):
            schema = cls.model_json_schema()
            properties = schema["properties"]
            json_format = "{\n"
            for field_name, field_info in properties.items():
                json_format += f'"{field_name}": "{field_info["define"]}"\n'
            json_format += "}"
            return json_format

    return DomInfo


def create_suggestion_dom(short_prompt, mid_prompt, long_prompt,
                          reflection_prompt):

    class DomInfo(BaseModel):
        if len(short_prompt) > 0:
            short_memory_index: str = Field(
                ...,
                description="短期记忆的索引ID,格式必须为:'S1,S2,S3',如果没有引用，则不返回 不能随意添加",
                define="S1,S2,S3",
            )
        if len(mid_prompt) > 0:
            mid_memory_index: str = Field(
                ...,
                description="中期记忆的索引ID,格式必须为:'M1,M2,M3',如果没有引用，则不返回 不能随意添加",
                define="M1,M2,M3",
            )
        if len(long_prompt) > 0:
            long_memory_index: str = Field(
                ...,
                description="长期记忆的索引ID,格式必须为:'L1,L2,L3',如果没有引用，则不返回 不能随意添加",
                define="L1,L2,L3",
            )
        if len(reflection_prompt) > 0:
            reflection_memory_index: str = Field(
                ...,
                description="过去反思记忆的索引ID,格式必须为:'R1,R2,R3',如果没有引用，则不返回 不能随意添加",
                define="R1,R2,R3",
            )
        summary_reason: str = Field(
            ...,
            description="推理细节, 根据专业交易员的交易建议，您能否向我详细解释为什么会根据您提供的信息做出这样的决定",
            define="string",
        )

        @classmethod
        def dumps(cls):
            schema = cls.model_json_schema()
            properties = schema["properties"]
            json_format = "{\n"
            for field_name, field_info in properties.items():
                json_format += f'"{field_name}": "{field_info["define"]}"\n'
            json_format += "}"
            return json_format

    return DomInfo
