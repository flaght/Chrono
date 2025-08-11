from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal


class Factor(BaseModel):
    name: str  ## 因子名称
    value: float  ## 因子值
    desc: str  ## 因子描述


class FactorsList(BaseModel):
    date: str  ## 日期
    desc: str  ## 描述
    ## 动态存储因子 键是因子ID
    factors: Dict[str, Factor] = Field(default_factory=dict)

    def text(self, include_head=False):
        texts = ""
        if include_head:
            texts += f"{self.desc} \n"
        texts += "{0} 数据值:\n".format(self.date)
        for factor_id, factor_data in self.factors.items():
            # 格式化输出，保留4位小数
            texts += f"{factor_data.name}: {factor_data.value:.4f}\n"
        return texts

    def desc_to_string(self):
        # 使用列表推导式高效地生成每一行
        desc_lines = [
            f"{factor.name}:{factor.desc}" for factor in self.factors.values()
        ]

        # 使用换行符将所有行连接成一个字符串
        return "\n".join(desc_lines)

    def markdown(self, include_value):
        if include_value:
            header = "| 因子名字 | 因子值 | 因子描述 |"
            separator = "| :--- | :--- | :--- |"
        else:
            header = "| 因子名字 | 因子描述 |"
            separator = "| :--- | :--- |"
        table_rows = [header, separator]

        # 遍历所有因子，为每个因子创建一行
        for factor in self.factors.values():
            if include_value:
                # 包含因子值，并格式化为4位小数
                row = f"| {factor.name} | {factor.value:.4f} | {factor.desc} |"
            else:
                # 不包含因子值
                row = f"| {factor.name} | {factor.desc} |"
            table_rows.append(row)

        # 使用换行符将所有行连接成一个完整的字符串
        return "\n".join(table_rows)


class FactorsGroup(BaseModel):
    date: str
    symbol: str
    index: Optional[str] = Field(default=None, nullable=True)
    factors_list: List[FactorsList] = Field(default_factory=list)

    def markdown(self, include_value):
        last_desc = ""
        texts = ""
        for factor_list in self.factors_list:
            current_desc = factor_list.markdown(include_value)
            if current_desc != last_desc:
                texts += current_desc
                texts += "\n"
        return texts

    def format(self, types: str = "short"):
        desc = {"short": "短期记忆", "mid": "中期记忆", "long": "长期记忆"}
        sidx = {"short": "S", "mid": "M", "long": "L"}

        texts = "**{0}索引ID**: {1}{2}\n".format(desc[types], sidx[types],
                                               self.index)
        for factor_list in self.factors_list:
            texts += factor_list.text(include_head=True)
        return texts


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
        name: Optional[str] = Field(
            default="",  # 提供一个默认值 None
            description="所属策略",
            define="",
        )
        if len(short_prompt) > 0:
            short_memory_index: str = Field(
                default=None,  # 提供一个默认值 None
                description="短期记忆的索引ID,格式必须为:'S1,S2,S3',如果没有引用，则不返回 不能随意添加",
                define="S1,S2,S3",
            )
        if len(mid_prompt) > 0:
            mid_memory_index: Optional[str] = Field(
                default=None,  # 提供一个默认值 None
                description="中期记忆的索引ID,格式必须为:'M1,M2,M3',如果没有引用，则不返回 不能随意添加",
                define="M1,M2,M3",
            )
        if len(long_prompt) > 0:
            long_memory_index: Optional[str] = Field(
                default=None,  # 提供一个默认值 None
                description="长期记忆的索引ID,格式必须为:'L1,L2,L3',如果没有引用，则不返回 不能随意添加",
                define="L1,L2,L3",
            )
        if len(reflection_prompt) > 0:
            reflection_memory_index: Optional[str] = Field(
                default=None,  # 提供一个默认值 None
                description="过去反思记忆的索引ID,格式必须为:'R1,R2,R3',如果没有引用，则不返回 不能随意添加",
                define="R1,R2,R3",
            )
        summary_reason: str = Field(
            description=
            "推理细节, 根据专业交易员的交易建议，您能否向我详细解释为什么会根据您提供的信息做出这样的决定。尽可能包含推理逻辑，使用到有效的因子",
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
        name: Optional[str] = Field(
            default="",  # 提供一个默认值 None
            description="所属策略",
            define="",
        )

        if len(short_prompt) > 0:
            short_memory_index: Optional[str] = Field(
                default=None,  # 提供一个默认值 None
                description="短期记忆的索引ID,格式必须为:'S1,S2,S3',如果没有引用，则不返回 不能随意添加",
                define="S1,S2,S3",
            )
        if len(mid_prompt) > 0:
            mid_memory_index: Optional[str] = Field(
                default=None,  # 提供一个默认值 None
                description="中期记忆的索引ID,格式必须为:'M1,M2,M3',如果没有引用，则不返回 不能随意添加",
                define="M1,M2,M3",
            )
        if len(long_prompt) > 0:
            long_memory_index: Optional[str] = Field(
                default=None,  # 提供一个默认值 None
                description="长期记忆的索引ID,格式必须为:'L1,L2,L3',如果没有引用，则不返回 不能随意添加",
                define="L1,L2,L3",
            )
        if len(reflection_prompt) > 0:
            reflection_memory_index: Optional[str] = Field(
                default=None,  # 提供一个默认值 None
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
