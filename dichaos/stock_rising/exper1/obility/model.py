from pydantic import BaseModel, Field, TypeAdapter
from typing import List, Literal, Dict
import pdb, json, re



def clean_and_extract_json(response_content: str):
    """提取并清理LLM响应中的JSON内容"""
    try:
        # 尝试直接解析
        return json.loads(response_content)
    except json.JSONDecodeError:
        # 使用正则匹配JSON结构
        match = re.search(r"({.*}|\[.*\])", response_content, re.DOTALL)
        if not match:
            return None
        try:
            # 清理常见错误（如末尾逗号）
            clean_content = re.sub(r",\s*([}\]])", r"\1", match.group(0))
            # 处理转义字符
            clean_content = clean_content.replace("\\n",
                                                  " ").replace("\n", " ")
            return json.loads(clean_content)
        except json.JSONDecodeError:
            return None


## 金融实际情况
class MarketOverview(BaseModel):
    code: str  ## 股票代码
    limit: Literal[0, 1]  ## 涨停标识 1 涨停  0 未涨停
    chg: float  ## 涨停幅度 5cm 10cm 20cm


class MarketOverviewList(BaseModel):
    date: str  ## 日期
    stocks: List[MarketOverview]

    def to_json(self):
        res = {}
        for stock in self.stocks:
            res[stock.code] = {
                "limit": u"涨停" if stock.limit == 1 else u"未涨停",
                u"涨跌幅": "{0}%".format(stock.chg * 100)
            }
        return res


## 技术指标
class Indicator(BaseModel):
    name: str  ## 指标名称
    id: str  ## 显示
    values: float  ## 指标值


class IndicatorSets(BaseModel):
    sma5: Indicator = Field(default=None, nullable=True)
    sma10: Indicator = Field(default=None, nullable=True)
    sma20: Indicator = Field(default=None, nullable=True)
    ema12: Indicator = Field(default=None, nullable=True)
    ema26: Indicator = Field(default=None, nullable=True)
    rsi: Indicator = Field(default=None, nullable=True)
    #macd: Indicator = Field(default=None, nullable=True)
    #atr: Indicator = Field(default=None, nullable=True)
    vwap: Indicator = Field(default=None, nullable=True)
    #adx: Indicator = Field(default=None, nullable=True)
    #obv: Indicator = Field(default=None, nullable=True)
    pp: Indicator = Field(default=None, nullable=True)
    r1: Indicator = Field(default=None, nullable=True)
    s1: Indicator = Field(default=None, nullable=True)
    r2: Indicator = Field(default=None, nullable=True)
    s2: Indicator = Field(default=None, nullable=True)
    r3: Indicator = Field(default=None, nullable=True)
    s3: Indicator = Field(default=None, nullable=True)
    code: str  ## 股票代码
    date: str  ## 日期

    def to_json(self):
        return {
            'sma5': self.sma5.values,
            'sma10': self.sma10.values,
            'sma20': self.sma20.values,
            'ema12': self.ema12.values,
            'ema26': self.ema26.values,
            'rsi': self.rsi.values,
            'vwap': self.vwap.values,
            'pp': self.pp.values,
            'r1': self.r1.values,
            's1': self.s1.values,
            'r2': self.r2.values,
            's2': self.s2.values,
            'r3': self.r3.values,
            's3': self.s3.values,
        }


## K线
class KLine(BaseModel):
    date: str
    code: str
    open: float = Field(default=None, nullable=True)
    close: float = Field(default=None, nullable=True)
    high: float = Field(default=None, nullable=True)
    low: float = Field(default=None, nullable=True)
    volume: float = Field(default=None, nullable=True)
    amount: float = Field(default=None, nullable=True)

    def to_json(self):
        return {
            "open": self.open,
            "close": self.close,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "amount": self.amount
        }


class IndicatorList(BaseModel):
    code: str
    date: str
    indicator: IndicatorSets
    kline: KLine

    def to_json(self, index):
        return {
            "index": index,
            "技术指标": self.indicator.to_json(),
            "K线数据": self.kline.to_json()
        }


def clean_and_extract_json(response_content: str):
    """提取并清理LLM响应中的JSON内容"""
    try:
        # 尝试直接解析
        return json.loads(response_content)
    except json.JSONDecodeError:
        # 使用正则匹配JSON结构
        match = re.search(r"({.*}|\[.*\])", response_content, re.DOTALL)
        if not match:
            return None
        try:
            # 清理常见错误（如末尾逗号）
            clean_content = re.sub(r",\s*([}\]])", r"\1", match.group(0))
            # 处理转义字符
            clean_content = clean_content.replace("\\n",
                                                  " ").replace("\n", " ")
            return json.loads(clean_content)
        except json.JSONDecodeError:
            return None


def create_suggestion_dom(short_prompt, mid_prompt, long_prompt,
                          reflection_prompt):

    class DomInfo(BaseModel):
        code: str = Field(
            ...,
            description="对应ticker或者code代码",
            define="",
        )
        short_memory_index: str = Field(
            ...,
            description="短期记忆的索引ID,格式必须为:'S1,S2,S3',如果没有引用，则不返回 不能随意添加",
            define="S1,S2,S3",
        )
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

    class DomInfos(BaseModel):
        tickers: List[DomInfo] = Field(
            ...,
            description="每支股票推理解释的原因",
            define="string",
        )
        details: str = Field(
            ...,
            description="根据各股票推理的原因，提取归纳总结出原因，越详细越好",
            define="string",
        )

        @classmethod
        def loads(cls, result):
            pdb.set_trace()
            cleaned = clean_and_extract_json(result.content)
            adapter = TypeAdapter(cls)

    return DomInfos
