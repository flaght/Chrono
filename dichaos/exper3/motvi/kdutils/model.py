from pydantic import BaseModel, Field


class Factor(BaseModel):
    name: str  ## 因子名称
    value: float  ## 因子值
    desc: str  ## 因子描述


class KLine(BaseModel):
    date: str
    symbol: str
    open: float = Field(default=None, nullable=True)
    close: float = Field(default=None, nullable=True)
    high: float = Field(default=None, nullable=True)
    low: float = Field(default=None, nullable=True)
    volume: float = Field(default=None, nullable=True)
    amount: float = Field(default=None, nullable=True)
    openint: float = Field(default=None, nullable=True)

    def format(self):
        text = "{0} K线数据:\n".format(self.date)
        text += "开盘价: {0}\n".format(self.open)
        text += "收盘价: {0}\n".format(self.close)
        text += "最高价: {0}\n".format(self.high)
        text += "最低价: {0}\n".format(self.low)
        text += "成交量: {0}\n".format(self.volume)
        text += "成交额: {0}\n".format(self.amount)
        text += "持仓量: {0}\n".format(self.openint)
        return text


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
            description="推理细节, 根据专业交易员的交易建议，您能否向我详细解释为什么会根据您提供的信息做出这样的决定。尽可能包含推理逻辑，使用到有效的因子",
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
