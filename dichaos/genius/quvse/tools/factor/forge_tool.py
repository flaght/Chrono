import pdb, os
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple, List
from langchain_core.messages import HumanMessage
from dichaos.services.react.tool import BaseTool, ToolResult
from dichaos.services.llm import LLMServiceFactory, MessageGenerator
from dichaos.env import DEBUG_MODE
from tools.factor.tool_generator import ToolGenerator

FORGE_SYSTEM_PROMPT = """
你是一位顶级的**量化因子发明家与构建专家**。你的任务是执行一个完整的、从零到多的因子创造流程。你将基于一个给定的工具箱，首先构思出一个核心的、创新的因子思路，然后将这个思路发散成一系列具体的、多样化的因子表达式。

你的工作流程分为两个严格的阶段：

---
### **阶段一：核心因子思路的构思 (DSL - The Core Idea)**

在这个阶段，你需要扮演**因子发明家**的角色。

**1. 任务:**
   *   审视用户提供的**有限工具箱**（特征和算子）。
   *   通过强大的**经济学直觉**和**逻辑推理**，主动构思一个可以利用这些工具构建的、具有潜在盈利能力的**核心Alpha因子思路**。

**2. 核心原则:**
   *   **工具驱动**: 你的构思必须源于给定的工具。
   *   **逻辑自洽**: 你的思路必须有一个清晰的市场假说。
   *   **组合创新**: 你的思路应该体现出工具之间新颖、有意义的组合方式。

---
### **阶段二：因子变体的展开 (EXPS - The Variations)**

在这个阶段，你需要扮演**因子构建专家**的角色，并将**阶段一构思出的核心思路**作为输入。

**1. 任务:**
   *   基于你的核心思路，创造性地生成用户要求数量的、**本质不同**的因子变体。
   *   **严禁只修改参数**；必须在结构、使用算子、或逻辑角度上有所创新。

**2. 核心原则:**
   *   **多样性**: 必须从多个不同维度进行发散性思考（不同指标组合、逻辑变形、非对称性、标准化等）。
   *   **严格合规**:
      *   所有表达式**必须且只能**使用第一阶段定义的工具箱。
      *   表达式必须是**单行、完整**的，严禁使用`let`、分号或任何不允许的语法。
      *   表达式建议使用的算子不超过6个，注重逻辑深度。
   *   **深度分析与绝对评分**: 为每个变体提供详细的解释、原理，并根据以下**绝对逻辑性标准**进行0-10分的评分：
      *   **高分 (8-10)**: 逻辑链条清晰、严谨，理论依据扎实，能高度、巧妙地实现核心思路。
      *   **中等分 (5-7)**: 逻辑基本合理，但可能存在一些较强的假设，或是对核心思路的常规实现。
      *   **低分 (0-4)**: 逻辑存在明显缺陷、过于牵强，或与核心思路关联性不强。

---
### **最终输出格式 (Final Output Format)**

你的最终输出**必须是、且只能是**一个包含`factors`列表的JSON对象。列表中应包含用户要求数量的因子。

注意 文本内容必须中文回复
"""

EXPS_USER_PROMPT = """
请严格遵循系统设定的两阶段工作流程，完成一次完整的因子创造任务。

---
### **输入：你的专属工具箱 (Your Toolbox)**

你本次设计**只能使用**以下元素：

*   **可用基础特征 (Base Features):**
    > `open`, `close`, `high`, `low`,  `volume`, `openint`

*   **可用用户指定特征 (User-Provided Features, if any):**
    > {user_features}

*   **可用算子 (Operators):**
    > {operators}

*  **表达式约束
  > 特征必须带引号. 如：SUBBED('bid_ask_spread', MA(5, 'bid_ask_spread'))
  > 要严格使用提供算子markdown表里的Expression字段信息，切勿将 其他信息使用。如将 ADDED生成ADD
  > 优先选择时序算子,其次再考虑基础算子
  

---
### **任务：执行完整的因子创造流程**

**1. [思维链 - 阶段一] 构思核心思路:**
   *   **(内心思考)** 首先，请在内部进行头脑风暴，基于上述工具箱构思一个核心的Alpha因子思路。这个思路应包含一个清晰的假说和实现该假说的基本逻辑描述。**这部分思考过程不需要直接输出。**

**2. [思维链 - 阶段二] 展开为多样化变体:**
   *   **(核心任务)** 现在，将你刚刚构思出的**核心思路**作为种子，生成一个逻辑性分数最好的前 **{factor_count}** 个**结构不同、逻辑相关性低**的因子变体的JSON输出。
   *   确保每个变体都严格使用了工具箱中的元素，并体现了高度的创造性和逻辑性。

**请直接生成最终的JSON输出:**
{{

 "factors":[     
 {{
        "expression": "str"
        "explanation": "str",
        "principle":"str",
        "score": "int"
  }},
  {{
        "expression": "str"
        "explanation": "str",
        "principle":"str",
        "score": "int"
  }}
  ],
  "dsl":{{
    "hypothesis": "<在这里填写你基于工具箱构思出的假说>",
    "description": "<在这里填写因子设计的简要描述>"
}}
}}
"""

tool_generator = ToolGenerator()


class DomInfo1(BaseModel):
    expression: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="因子表达式",
        define="",
    )
    explanation: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="详细的逻辑解释",
        define="",
    )
    principle: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="因子背后的理论依据",
        define="",
    )
    score: Optional[float] = Field(
        default="",  # 提供一个默认值 None
        description="<综合评分>",
        define="",
    )


class DomInfo2(BaseModel):
    hypothesis: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="你的假-说",
        define="",
    )
    description: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="你的因子描述",
        define="",
    )


class DomInfos3(BaseModel):
    factors: List[DomInfo1] = Field(
        ...,
        description="每个推理的因子详情",
        define="string",
    )

    dsl: Optional[DomInfo2] = Field(
        default="",  # 提供一个默认值 None
        description="你的设计描述",
        define="",
    )


class FactorGenerator(object):

    def __init__(self, llm_provider, llm_model):
        self.llm_service = LLMServiceFactory.create_llm_service(
            llm_provider=llm_provider,
            llm_model=llm_model,
            system_message=FORGE_SYSTEM_PROMPT,
            other_parameters={'temperature': 0.0})

        self.message_generator = MessageGenerator(llm_service=self.llm_service,
                                                  debug_mode=DEBUG_MODE)

    async def generate(self, features: Optional[str], operators: Optional[str],
                       count: Optional[int]) -> Dict[str, str]:
        content = await self.message_generator.generate_message_async(
            messages=[
                HumanMessage(content=EXPS_USER_PROMPT.format(
                    **{
                        'user_features': features,
                        'operators': operators,
                        'factor_count': count
                    }))
            ],
            output_schema=DomInfos3)
        return content.model_dump()


class ForgeFactorTool(BaseTool):
    name: str = "factor_forge_tool"
    description: str = "擅长从基础工具集（特征与算子）中推理演绎多个因子表达式和描述"
    parameters: dict = {
        "type": "object",
        "properties": {
            "count": {
                "type": "string"
            },
            "category": {
                "type": "string"
            }
        },
        "required": ["features"]
    }
    _factor_generator: Optional[FactorGenerator] = None

    def _initialize(self):
        if self._factor_generator is None:
            self._factor_generator = FactorGenerator(
                llm_model=os.environ['MODEL_NAME'],
                llm_provider=os.environ['MODEL_PROVIDER'])

    async def execute(self, category: str, count: str) -> ToolResult:
        try:
            self._initialize()
            tools = tool_generator.create_tools(category=category, k=count)
            expr = await self._factor_generator.generate(
                count=count,
                features=tools['features'],
                operators=tools['expression'])
            return ToolResult(output={"expression": expr})
        except Exception as e:
            return ToolResult(error=f"表达式生成失败: {e}")
