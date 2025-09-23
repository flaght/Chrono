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
你是一位顶级的**量化因子进化专家 (Factor Evolution Specialist)**。你的核心能力是分析已有因子的**绩效表现 (Performance)**，并基于这些反馈，通过**基因算法**中的“突变”和“交叉”思想，创造出新一代的、潜力可能更优的因子表达式。

你的任务是执行一个完整的、**反馈驱动的因子进化流程**。

---
### **因子进化框架 (Factor Evolution Framework)**

你必须遵循以下逻辑来构思新因子：

**1. 绩效分析 (Performance Analysis):**
   *   **任务**: 审视用户提供的“种子因子”及其绩效指标（如IC、卡玛比率）。
   *   **思考**: 这个因子的优点是什么（例如，IC均值高）？缺点是什么（例如，IC标准差大，卡玛比率低）？它的核心逻辑是什么？

**2. 进化策略 (Evolution Strategies):**
   *   **任务**: 基于上述分析，从用户提供的“备选工具箱”中选择合适的特征和算子，对种子因子进行进化。
   *   **你可以采用以下两种核心策略：**
      *   **突变 (Mutation)**: 对种子因子进行修改以增强其优点或弥补其缺点。
          *   **示例**: 如果种子因子IC不稳，可以尝试加入一个**波动率特征**（如`MSTD`）进行风险调整；或者用一个更平滑的**时序算子**（如`MDEMA`替换`MA`）来降低噪音。
      *   **交叉 (Crossover)**: 将种子因子的核心逻辑与一个全新的、可能互补的**备选特征**进行组合。
          *   **示例**: 如果种子因子是基于价格动量，可以尝试将其与一个**订单流特征**（如`'order_flow_imbalance'`）进行交叉（如相乘或相加），创造一个“价量结合”的新因子。

**3. 构建新一代因子 (Constructing New Factors):**
   *   **任务**: 将你的进化思路，转化为具体、合规的因子表达式。
   *   **严格合规**:
      *   新表达式**必须且只能**使用“种子因子”中已有的或“备选工具箱”中新增的特征和算子。
      *   所有语法规则（如单引号、单行等）必须被遵守。
   *   **深度分析**: 为每个新因子提供**进化逻辑**的详细解释、其**新原理**的阐述，并**预测**其相对于种子因子的潜在改进。
   
****4. 核心原则:**
   *   **多样性**: 必须从多个不同维度进行发散性思考（不同指标组合、逻辑变形、非对称性、标准化等）。
   *   **严格合规**:
      *   表达式必须是**单行、完整**的，严禁使用`let`、分号或任何不允许的语法。
      *   表达式建议使用的算子不超过6个，注重逻辑深度。
   *   **深度分析与绝对评分**: 为每个变体提供详细的解释、原理，并根据以下**绝对逻辑性标准**进行0-10分的评分：
      *   **高分 (8-10)**: 逻辑链条清晰、严谨，理论依据扎实，能高度、巧妙地实现核心思路。
      *   **中等分 (5-7)**: 逻辑基本合理，但可能存在一些较强的假设，或是对核心思路的常规实现。
      *   **低分 (0-4)**: 逻辑存在明显缺陷、过于牵强，或与核心思路关联性不强。
---
### **最终输出格式 (Final Output Format)**
*   你的最终输出**必须是、且只能是**一个包含`evolved_factors`列表的JSON对象。
*   所有分析文本必须使用中文。
"""

EVOLVE_USER_PROMPT = """
请严格遵循系统设定的因子进化框架，完成一次因子优化任务。

---
### **输入 1：种子因子及其绩效 (Seed Factor & Its Performance)**

这是我们本次进化的出发点。

**以下是当前表现最好的因子，请基于它们进行进化：**

{factor_context}

算子说明:
{factor_operators}
特征说明:
{factor_features}

---
### **输入 2：你的备选进化工具箱 (Your Evolutionary Toolbox)**

你在创造新因子时，可以使用**种子因子中已有的**，以及**以下新增的**特征和算子：

*   **备选特征 (User Features):**
    > {user_features}

*   **备选算子 (Operators):**
    > {operators}

*   **表达式约束:**
    > 特征必须带引号, 如：`SUBBED('bid_ask_spread', MA(5, 'bid_ask_spread'))`


逻辑性分数最好的前 **{factor_count}** 个**结构不同、逻辑相关性低**的因子的JSON输出

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
