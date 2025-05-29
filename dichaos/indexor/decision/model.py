from pydantic import BaseModel, Field
from typing_extensions import Literal


class TraderSignal(BaseModel):
    name: str = Field(
        ...,
        description="交易信号的名称, 请提供详细的解释,而且必须返回内容",
        define="string",
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

    def format(self):
        text = "策略交易名称:{0}\n 策略交易理由:{1}\n 置信度:{2}\n 交易信号:{3}\n 分析细节:{4}\n\n\n".format(
            self.name, self.reasoning, self.confidence, self.signal,
            self.analysis_details)
        return text


class Memory(BaseModel):
    date: str
    symbol: str
    index: str = Field(default=None, nullable=True)
    response: str = Field(default=None, nullable=True)

    def format(self, types: str = "short"):
        desc = {"short": "短期记忆"}
        sidx = {"short": "S"}

        text = "{0}索引ID: {1}{2}\n".format(desc[types], sidx[types], self.index)
        text += self.response
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
            description="根据专业交易员的交易建议，您能否向我解释为什么交易员会根据您提供的信息做出这样的决定",
            define="string",
        )

        analysis_details: str = Field(
            ...,
            description="分析评估各子策略的交易信号的有效性，准确性，贡献度等",
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
            description="交易动作, 请提供详细的解释,而且必须返回内容",
            define="string",
        )

        action: str = Field(
            ...,
            description="基于市场数据和分析, 请做出交易动作",
            define="buy/sell/short/cover/hold",
        )

        quantity: int = Field(
            ...,
            description="交易数量",
            define="int",
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
