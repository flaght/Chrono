import pdb, os
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple, List
from langchain_core.messages import HumanMessage
from dichaos.services.react.tool import BaseTool, ToolResult
from dichaos.services.llm import LLMServiceFactory, MessageGenerator
from dichaos.env import DEBUG_MODE
from tools.rulex.tool_generator import ToolGenerator

FORGE_SYSTEM_PROMPT = """
你是一位顶级的**全栈量化策略设计师**。你的任务是从最基础的构建块（特征、算子、信号函数、持仓函数）出发，**自主地、从零开始**，设计并构建一套完整的、可执行的、逻辑严谨的交易策略。

你的工作是执行一个**端到端的策略创造流程**，最终输出一个包含所有必要组件的结构化策略定义。

---
### **策略构建的思维链 (Chain of Thought for Strategy Creation)**

你必须严格遵循以下步骤来构建你的策略：

**步骤 1: 构思核心信号公式 (`formual`)**
*   **任务**: 审视用户提供的**基础特征**和**基础算子**，通过强大的经济学直觉和逻辑推理，创造性地组合出一个核心的Alpha信号公式。
*   **原则**: 这个公式应该具备创新性、逻辑自洽性，并且是后续所有策略组件的基础。
*   **规则**: 公式必须使用4个算子以上，其中必须优先考虑时序算子，每个公式中必须包含两个以上的时序算子
*   **独立性**:  公式

**步骤 2: 设计信号处理方法 (`signal_method` & `signal_params`)**
*   **任务**: 审视用户提供的**基础信号函数**列表。根据你在步骤1中构思出的`formual`的特性（例如，是趋势型还是回归型？数值范围？），选择一个**最匹配**的信号处理函数。
*   **原则**: 你的选择必须有充分的理由。例如，对于一个噪音较大的原始信号，选择`mean_signal`（均值平滑）是合理的。同时，为该函数提供一套合理的初始参数。

**步骤 3: 设计头寸与风险管理方法 (`strategy_method` & `strategy_params`)**
*   **任务**: 审视用户提供的**基础持仓函数**列表。根据整个策略的交易逻辑（例如，是想跟随趋势还是在区间内交易？），选择一个**最匹配**的头寸与风险管理函数。
*   **原则**: 你的选择必须与信号的特性相辅相成。例如，对于一个动量型信号，选择`trailing_atr_strategy`（ATR移动止损）是合理的。同时，为该函数提供一套合理的初始参数。

**步骤 4: 撰写分析与解释**
*   **任务**: 为你从零构建的这套完整策略，提供深入的分析，包括：
    *   **`explanation`**: 详细解释这四个核心组件 (`formual`, `signal_method`, `strategy_method`, `params`)是如何协同工作的，以及做出每一项选择的完整理由。
    *   **`principle`**: 必须总结这个完整策略背后的核心交易思想或市场假说。
    *   **`score`**: 根据策略的逻辑一致性、创新性和完整性，给出一个0-10分的绝对评分，分数必须在0~10之间，可以是保留一位小数
---
    
### **最终输出格式 (Final Output Format)**

你的最终输出**必须是、且只能是**一个包含了上述所有策略组件和分析的JSON对象。所有文本内容必须使用中文回复。
"""

EXPS_USER_PROMPT = """

请严格遵循系统设定的思维链，基于以下给定的**完整工具箱**，从零开始构建一套完整的交易策略。

---
### **输入：你的完整工具箱 (Your Full Toolbox)**

你本次设计**只能使用**以下元素：

*   **可用基础特征 (Base Features):**
    > `open`, `close`, `high`, `low`,  `volume`, `openint`

*   **可用用户指定特征 (User-Provided Features, if any):**
    > {user_features}

*   **可用基础算子 (Base Operators):**
    > {operators}

*   **可用基础信号函数 (Base Signal Functions):**
    > {signal_functions}

*   **可用基础持仓函数 (Base Strategy Functions):**
    > {strategy_functions}

*   **表达式约束:**
    > 特征必须带引号, 如：`SUBBED('bid_ask_spread', MA(5, 'bid_ask_spread'), EMA(5,'bid_ask_spread'))`
    > 特征名称必须可用基础特征和可用用户指定特征，不得做其他任何改变
    > 表达式之间低相关性，包括特征低相关性，算子低相关性

---
### **任务：执行完整的策略构建流程**

**1. [思维链]**:
   *   **(内心思考)** 首先，请在内部进行完整的四步思考：构思`formual` -> 选择`signal_method` -> 选择`strategy_method` -> 设定所有`params`。

**2. [核心任务]**:
   *   **(最终输出)** 将你的完整设计，必须生成一个逻辑性分数最好的前 **{factor_count}** 个结构不同、逻辑相关性低的策略变体的JSON输出。
   *   **对每个变体进行独立的、绝对的评分，并将最终的策略列表按`score`从高到低进行排序。**
   *   确保每个变体都严格使用了工具箱中的元素，并体现了高度的创造性和逻辑性。

**请直接生成最终的JSON输出:**
```json
{{

 "strategy":[     
 {{
        "expression": "str (你基于特征和算子自主构思的信号公式)",
        "strategy_method": "str (从基础持仓函数列表中选择)",
        "strategy_params": {{
            "param1": "value1",
            "param2": "value2"
        }},
        "signal_method": "str (从基础信号函数列表中选择)",
        "signal_params": {{
            "param1": "value1",
            "param2": "value2"
        }},
        "explanation": "str",
        "principle": "str",
        "score": "int"
  }},
  {{
        "expression": "str (你基于特征和算子自主构思的信号公式)",
        "strategy_method": "str (从基础持仓函数列表中选择)",
        "strategy_params": {{
            "param1": "value1",
            "param2": "value2"
        }},
        "signal_method": "str (从基础信号函数列表中选择)",
        "signal_params": {{
            "param1": "value1",
            "param2": "value2"
        }},
        "explanation": "str",
        "principle": "str",
        "score": "int"
  }}
  ],
  "dsl":{{
    "hypothesis": "<在这里填写你基于工具箱构思出的假说>",
    "description": "<在这里填写策略设计的简要描述>"
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

    signal_method: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="信号函数",
        define="",
    )

    signal_params: Optional[dict] = Field(
        default="",  # 提供一个默认值 None
        description="信号函数参数",
        define="",
    )

    strategy_method: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="持仓函数",
        define="",
    )

    strategy_params: Optional[dict] = Field(
        default="",  # 提供一个默认值 None
        description="持仓函数参数",
        define="",
    )

    explanation: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="详细的逻辑解释",
        define="",
    )
    principle: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="策略背后的理论依据",
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
        description="你的策略描述",
        define="",
    )


class DomInfos3(BaseModel):
    strategy: List[DomInfo1] = Field(
        ...,
        description="每个推理的策略详情",
        define="string",
    )

    dsl: Optional[DomInfo2] = Field(
        default="",  # 提供一个默认值 None
        description="你的设计描述",
        define="",
    )


class RulexGenerator(object):

    def __init__(self, llm_provider, llm_model):
        self.llm_service = LLMServiceFactory.create_llm_service(
            llm_provider=llm_provider,
            llm_model=llm_model,
            system_message=FORGE_SYSTEM_PROMPT,
            other_parameters={'temperature': 0.0})

        self.message_generator = MessageGenerator(llm_service=self.llm_service,
                                                  debug_mode=DEBUG_MODE)

    async def generate(self, features: Optional[str], operators: Optional[str],
                       signal_functions: Optional[str],
                       strategy_functions: Optional[str],
                       count: Optional[int]) -> Dict[str, str]:
        content = await self.message_generator.generate_message_async(
            messages=[
                HumanMessage(content=EXPS_USER_PROMPT.format(
                    **{
                        'user_features': features,
                        'operators': operators,
                        'signal_functions': signal_functions,
                        'strategy_functions': strategy_functions,
                        'factor_count': count
                    }))
            ],
            output_schema=DomInfos3)
        return content.model_dump()


class RucgeRulexTool(BaseTool):
    name: str = "rulex_rucge_tool"
    description: str = "擅长从基础工具集（特征,算子,信号函数，持仓函数）中推理演绎多个策略表达式和描述"
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
        "required": ["count", "category"]
    }
    _factor_generator: Optional[RulexGenerator] = None

    def _initialize(self):
        if self._factor_generator is None:
            self._factor_generator = RulexGenerator(
                llm_model=os.environ['MODEL_NAME'],
                llm_provider=os.environ['MODEL_PROVIDER'])

    async def execute(self, category: str, count: str) -> ToolResult:
        try:
            self._initialize()
            tools = tool_generator.create_tools(category=category, k=count)
            expr = await self._factor_generator.generate(
                count=count,
                features=tools['features'],
                signal_functions=tools['signal'],
                strategy_functions=tools['holding'],
                operators=tools['expression'])
            return ToolResult(output={"expression": expr})
        except Exception as e:
            return ToolResult(error=f"表达式生成失败: {e}")
