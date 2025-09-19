import pdb, os
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple
from langchain_core.messages import HumanMessage
from dichaos.services.react.tool import BaseTool, ToolResult
from dichaos.services.llm import LLMServiceFactory, MessageGenerator
from dichaos.env import DEBUG_MODE
from utils.expression import expression_str_md


features_str_md = """
    `$open`: 开盘价, 
    `$high`: 最高价, 
    `$low`: 最低价, 
    `$close`: 收盘价, 
    `$volume`: 成交量, 
    `$openint`:持仓量,
    `money_flow_volume_in`: 主动买入的总成交量,
    `money_flow_volume_out`: 主动卖出的总成交量,
    `vwap_tick`:`单笔成交均价`,
    `vwap_total`:时间窗⼝总成交均价,
    `money_flow_tick_in`:主动买⼊的成交笔数
    `money_flow_net_tick_in`:净主动买⼊笔数,
    `money_flow_net_volume_in`:净主动买⼊成交量
    `money_flow_smart_money_out`:聪明钱卖出⾦额（⾼于均价的主动卖单）
"""

DSL_SYSTEM_PROMPT = """
你是一位顶级的**量化因子发明家 (Factor Inventor)**，一个擅长从**基础工具集（特征与算子）**中挖掘深刻市场逻辑并构建全新Alpha因子的AI专家。你的核心能力是将看似孤立的构建块，通过强大的**经济学直觉**和**逻辑推理**，组合成有意义且具备创新性的因子。

你的核心任务是：审视用户提供的**有限工具箱**，主动构思一个可以利用这些工具构建的、具有潜在盈利能力的Alpha因子提案。

**你的提案必须遵循以下四大原则：**
1.  **工具驱动 (Tool-Driven):** 你的整个因子构思**必须源于**给定的特征和算子。你的假说和设计必须是这个工具箱能**合乎逻辑地**表达出来的。
2.  **逻辑自洽 (Logically Consistent):** 你必须为你的因子设计提供一个清晰的市场假说，解释为什么这个**特定的工具组合**能够捕捉到一种市场行为或异象。
3.  **组合创新 (Combinatorial Innovation):** 你的价值在于发现这些基础工具之间**新颖的、有意义的组合方式**。避免那些最显而易见、平庸的组合。
4.  **明确引用 (Explicit Referencing):** 在你的因子描述中，当提到任何特征或算子时，必须**明确使用其给定的标识符**，例如：“计算收盘价(`$close`)的5周期加权移动平均(`WMA`)”。
"""

DSL_USER_PROMPT = """
作为一名经验丰富的量化因子发明家，你的任务是基于以下给定的**工具箱**，**从零开始**构思并提出一个全新的Alpha因子。

---
### **第一部分：你的工具箱 (Your Toolbox)**

你本次设计**只能使用**以下元素：

*   **可用特征 (Features) 及其标识符:**
    > {features}

*   **可用算子 (Operators) 及其标识符:**
    > {operators}

---
### **第二部分：你的任务 (Your Mission)**

**1. 头脑风暴 (Brainstorming):**
*   审视你的工具箱。这些工具的组合能用来衡量市场的什么现象？（例如：动量？反转？波动性？价量关系？）
*   你能否构建一个捕捉**投资者非理性行为**（如恐慌、贪婪）或**市场微观结构**特征的逻辑？

**2. 核心假说构建 (Hypothesis Formulation):**
*   基于你的头脑风暴，提炼出一个核心的、可盈利的市场假说。这个假说必须清晰地解释为什么你设计的**工具组合**是有效的。

**3. 因子量化设计 (Factor Specification):**
*   详细描述如何一步步使用**工具箱中的标识符**来实现你的假说。
*   你的描述应清晰地展现出设计的**逻辑性**和**组合的创新性**。

---
### **第三部分：最终输出 (Final Output)**
请将你的**假说**和**因子设计**严格按照以下标准JSON格式返回。禁止使用 被```json  ```包裹的方式 

{{
    "hypothesis": "<在这里填写你基于工具箱构思出的假说>",
    "description": "<在这里填写因子设计的简要描述>"
}}
"""


class DomInfo(BaseModel):
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


class IdeaGenerator(object):

    def __init__(self, llm_provider, llm_model):
        self.llm_service = LLMServiceFactory.create_llm_service(
            llm_provider=llm_provider,
            llm_model=llm_model,
            system_message=DSL_SYSTEM_PROMPT,
            other_parameters={'temperature': 1.0})

        self.message_generator = MessageGenerator(llm_service=self.llm_service,
                                                  debug_mode=DEBUG_MODE)

    async def generate(self) -> Dict[str, str]:
        content = await self.message_generator.generate_message_async(
            messages=[
                HumanMessage(content=DSL_USER_PROMPT.format(
                    **{
                        'features': features_str_md,
                        'operators': features_str_md
                    }))
            ],
            output_schema=DomInfo)
        return content.model_dump()


class DSLFactorTool(BaseTool):
    name: str = "factor_dsl_tool"
    description: str = "擅长从基础工具集（特征与算子）中推理设计因子。"
    parameters: dict = {
        "type": "object",
        "properties": {
            "context": {
                "type": "string"
            }
        },
        "required": ["context"]
    }
    _idea_generator: Optional[IdeaGenerator] = None

    def _initialize(self):
        if self._idea_generator is None:
            self._idea_generator = IdeaGenerator(
                llm_model=os.environ['MODEL_NAME'],
                llm_provider=os.environ['MODEL_PROVIDER'])

    async def execute(self, context: str) -> ToolResult:
        try:
            self._initialize()
            output = await self._idea_generator.generate()
            if not output.get("hypothesis"):
                return ToolResult(error="LLM未能生成有效的因子设想。")
            return ToolResult(output=output)
        except Exception as e:
            return ToolResult(error=f"因子设想失败: {e}")
