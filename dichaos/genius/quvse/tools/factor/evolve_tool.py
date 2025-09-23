import os, pdb, json
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple, List
from langchain_core.messages import HumanMessage
from dichaos.services.react.tool import BaseTool, ToolResult
from dichaos.services.llm import LLMServiceFactory, MessageGenerator
from dichaos.env import DEBUG_MODE
from tools.factor.tool_generator import ToolGenerator
from utils.parsers import generate_markdown_table

EVOLVE_SYSTEM_PROMPT = """
你是一位顶级的**量化因子进化专家 (Factor Evolution Specialist)**。你的核心能力是分析已有因子的表现，并通过基因算法思想，创造出一系列**结构各异、逻辑正交、低相关性**的新一代因子。

你的任务是执行一个**反馈驱动的、以多样性为目标的因子进化流程**。

---
### **因子进化框架 (Factor Evolution Framework)**

你必须遵循以下逻辑来构思新因子：

**1. 绩效与结构分析 (Performance & Structural Analysis):**
   *   **任务**: 审视用户提供的“种子因子”。
   *   **思考**: 这个因子的核心逻辑是什么？（例如：趋势？反转？价量关系？）它主要依赖哪些特征和算子？它的已知优缺点是什么？

**2. 核心进化目标：追求低相关性 (Core Goal: Seeking Low Correlation)**
   *   你生成的每一个新因子，都必须在**核心逻辑**或**使用的关键信息源（特征）**上与种子因子及其他新因子有显著区别。
   *   **严禁**只进行微小的调整（如改变周期参数、用`EMA`替换`MA`等），这些被视为**高相关性**的变体。

**3. 进化策略 (Evolution Strategies for Diversity):**
   *   **任务**: 基于分析，从“备选工具箱”中选择**不同的**新元素，对种子因子进行进化。
   *   **突变 (Structural Mutation)**: 对种子因子进行**结构性**修改。
       *   **示例**: 将一个线性关系（如`SUBBED`）替换为一个**非线性关系**（如`SIGMOID`或`RANK`）；引入一个**条件逻辑**。
   *   **交叉 (Orthogonal Crossover)**: 将种子因子的逻辑与一个**完全不同类型**的备选特征进行组合。
       *   **示例**: 如果种子因子是**价量类**的，尝试与一个**情绪类**或**基本面类**的备选特征进行交叉，而不是与另一个价量特征交叉。**每个新因子应尝试与一个不同的备选特征交叉**。
   *   **全新创造 (De Novo Creation)**: 如果备选工具箱足够丰富，可以尝试从备选特征和算子中，构建一个与种子因子逻辑完全不同，但可能同样有效的新因子。

**4. 构建与分析 (Construction & Analysis):**
   *   **任务**: 将你的进化思路转化为合规的因子表达式，并提供深度分析。
   *   **严格合规**:
      *   表达式必须单行、完整，并遵守所有语法规则。
      *   表达式建议使用的算子不超过6-8个，注重逻辑深度。
   *   **深度分析与评分 (0-10分)**:
      *   **高分 (8-10)**: 逻辑清晰，且**与种子因子相关性低**，提供了全新的Alpha视角。
      *   **中等分 (5-7)**: 逻辑合理，但在结构上仍与种子因子有部分相似性。
      *   **低分 (0-4)**: 逻辑存在缺陷，或与种子因子高度相关。

---
### **最终输出格式 (Final Output Format)**
*   你的最终输出**必须是、且只能是**一个包含`factors`列表和`dsl`对象的JSON。
*   所有分析文本必须使用中文。
"""


EVOLVE_USER_PROMPT = """
请严格遵循系统设定的、以**多样性和低相关性**为核心的因子进化框架，完成本次优化任务。

---
### **输入 1：种子因子列表 (Seed Factor List)**

这是我们本次进化的出发点。请分析这些因子的共性与特性。
```
{factor_context}
```

**算子说明:**
{factor_operators}
**特征说明:**
{factor_features}

---
### **输入 2：你的备选进化工具箱 (Your Evolutionary Toolbox)**

你在创造新因子时，可以使用**种子因子中已有的**，以及**以下新增的**特征和算子：

*   **备选特征 (Candidate Features):**
    > {user_features}

*   **备选算子 (Candidate Operators):**
    > {operators}

*   **表达式约束:**
    > 特征必须带引号, 如：`SUBBED('bid_ask_spread', MA(5, 'bid_ask_spread'))`

---
### **核心任务**

基于对种子因子列表的分析，利用备选工具箱，为我生成 **{factor_count}** 个**结构不同、逻辑相关性低**的新因子。
**最终的输出列表应优先考虑多样性，而不是仅仅追求最高分。**

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


class EvolveGenerator(object):

    def __init__(self, llm_provider, llm_model):
        self.llm_service = LLMServiceFactory.create_llm_service(
            llm_provider=llm_provider,
            llm_model=llm_model,
            system_message=EVOLVE_SYSTEM_PROMPT,
            other_parameters={'temperature': 0.0})

        self.message_generator = MessageGenerator(llm_service=self.llm_service,
                                                  debug_mode=DEBUG_MODE)

    async def generate(self, factor_context: Optional[str],
                       random_features: Optional[str],
                       random_operators: Optional[str],
                       factor_features: Optional[str],
                       factor_operators: Optional[str],
                       count: Optional[int]) -> Dict[str, str]:

        content = await self.message_generator.generate_message_async(
            messages=[
                HumanMessage(content=EVOLVE_USER_PROMPT.format(
                    **{
                        'factor_context': factor_context,
                        'user_features': random_features,
                        'operators': random_operators,
                        'factor_features': factor_features,
                        'factor_operators': factor_operators,
                        'factor_count': count
                    }))
            ],
            output_schema=DomInfos3)
        return content.model_dump()


class EvolveFactorTool(BaseTool):
    name: str = "factor_evolve_tool"
    description: str = "擅长分析已有因子的**绩效表现 (Performance)**，创造出新一代的、潜力可能更优的因子表达式"
    parameters: dict = {
        "type": "object",
        "properties": {
            "count": {
                "type": "string"
            },
            "category": {
                "type": "string"
            },
            "factor_context": {
                "type": "string"
            }
        },
        "required": ["count", "category"]
    }
    _evole_generator: Optional[EvolveGenerator] = None

    def _initialize(self):
        if self._evole_generator is None:
            self._evole_generator = EvolveGenerator(
                llm_model=os.environ['MODEL_NAME'],
                llm_provider=os.environ['MODEL_PROVIDER'])

    async def execute(self, factor_context: str, category: str,
                      count: str) -> ToolResult:
        try:
            self._initialize()
            factor_context = json.loads(factor_context)
            random_tools = tool_generator.create_tools1(category=category,
                                                       k=count)
            ##  算子特征提取
            expressions = [factor['expression'] for factor in factor_context]
            factor_tools = tool_generator.expressions_disassembly(
                category, expressions)
            expr = await self._evole_generator.generate(
                count=count,
                factor_context=factor_context,
                factor_features=factor_tools['features'],
                factor_operators=factor_tools['expression'],
                random_features=random_tools['features'],
                random_operators=random_tools['expression'])
            return ToolResult(output={"expression": expr})
        except Exception as e:
            return ToolResult(error=f"表达式生成失败: {e}")
