from motvi.kdutils.model import *
import pdb, json


class AgentsResult(BaseModel):
    symbol: str
    name: str ## agent 名称
    desc: str  ## agent描述
    date: str  ## 日期
    reasoning: str  ## 决策逻辑
    confidence: int  ## 置信度
    signal: str  ## 信号
    analysis_details: str  ## 关键特征分析

    def format(self, include_head=False, to_text=False):
        dis = {}
        if include_head:
            dis['trade_date'] = self.date
        dis['desc'] = self.desc
        dis['reasoning'] = self.reasoning
        dis['confidence'] = self.confidence
        dis['signal'] = self.signal
        dis['analysis_details'] = self.analysis_details
        return json.dumps(dis) if to_text else dis


class AgentsGroup(BaseModel):
    date: str
    symbol: str
    index: Optional[str] = Field(default=None, nullable=True)
    agents_list: List[AgentsResult] = Field(default_factory=list)

    def format(self, types: str = "short"):
        desc = {"short": "短期记忆", "mid": "中期记忆", "long": "长期记忆"}
        sidx = {"short": "S", "mid": "M", "long": "L"}
        texts = "{0}索引ID: {1}{2}\n".format(desc[types], sidx[types],
                                           self.index)

        res = []
        for result in self.agents_list:
            res.append(result.format())

        texts += json.dumps(res, ensure_ascii=False)
        return texts



def create_prediction_dom1(short_prompt, mid_prompt, long_prompt,
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