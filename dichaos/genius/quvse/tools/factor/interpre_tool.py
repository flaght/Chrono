import pdb, os, re
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple, List
from langchain_core.messages import HumanMessage
from dichaos.services.react.tool import BaseTool, ToolResult
from dichaos.services.llm import LLMServiceFactory, MessageGenerator
from dichaos.env import DEBUG_MODE
from tools.factor.tool_generator import ToolGenerator

INTERPRE_SYSTEM_PROMPT = """
你是一位顶级的**量化因子分析师**，精通拆解复杂的因子表达式，并能从经济学直觉和市场微观结构的角度，对其背后的逻辑和理论依据进行深度剖析。

你的**唯一任务**是：接收一个用户给定的**因子表达式**和相关的**工具箱（特征与算子说明）**，然后对该表达式进行全面的分析，并以一个结构化的JSON格式返回你的分析报告。

---
### **分析框架 (Analytical Framework)**

你必须严格遵循以下三步分析流程：

**步骤 1: 表达式拆解 (Expression Decomposition)**
*   **任务**: 将给定的`expression`从内到外、一步一步地拆解。
*   **要求**: 识别出每一个用到的**特征**和**算子**，并结合工具箱中的描述，解释它们在表达式中各自的作用。

**步骤 2: 逻辑与原理阐述 (Logic & Principle Elucidation)**
*   **任务**: 综合步骤1的拆解结果，构建一个完整的逻辑链条。
*   **要求**:
    *   **`explanation` (详细逻辑解释)**: 详细说明这个表达式**作为一个整体**是如何运作的。它在试图衡量市场的什么具体现象？（例如：动量、反转、价量背离、聪明钱行为等）。
    *   **`principle` (因子背后的理论依据)**: 阐述该因子能够盈利背后可能存在的理论依据或市场假说。它利用了何种市场异象或投资者行为偏差？
*  **铁律**：
    *  不要强行解释，不符合逻辑和市场规律，则直接说不符合逻辑

**步骤 3: 综合评分 (Comprehensive Scoring)**
*   **任务**: 对该因子的设计给出一个客观、绝对的评分。
*   **要求**:
    *   `score`**必须**是一个0.0到10.0之间的数字（可含一位小数）。
    *   **评分依据**:
        *   **高分 (8.0-10.0)**: 表达式逻辑清晰、创新性强，且理论依据扎实。
        *   **中等分 (5.0-7.9)**: 逻辑合理，但可能是对一些已知概念的常规实现。
        *   **低分 (0.0-4.9)**: 逻辑存在缺陷、难以解释，或过于简单/冗余。

---
### **最终输出格式 (Final Output Format)**
*   你的最终输出**必须是、且只能是**一个纯净的JSON对象。
*   所有分析文本必须使用中文。
"""

INTERPRE_USER_PROMPT = """
请严格遵循系统设定的分析框架，对以下给定的**因子表达式**进行深度分析。

---
### **输入 1：待分析的因子表达式 (Expression to Analyze)**

*   **`expression`**: `{expression}`

---
### **输入 2：你的参考工具箱 (Your Reference Toolbox)**

#### **可用特征 (Features) 及其说明:**
你**必须**使用这些特征的官方名称和描述来理解表达式。
| 标识符 (Identifier) | 描述 (Description) |
|---|---|
{user_features}

#### **可用算子 (Operators) 及其说明:**
你**必须**使用这些算子的官方名称和描述来理解表达式。
> {operators}

---
### **核心任务**

为上述表达式生成一份包含**详细逻辑解释**、**理论依据**和**综合评分**的JSON分析报告。

**请直接生成最终的JSON输出:**
```json
{{
  "explanation": "str (详细解释这个表达式如何运作，衡量什么现象)",
  "principle": "str (阐述该因子背后的市场假说或金融理论)",
  "score": "float (0.0 ~ 10.0 之间)"
}}
"""

tool_generator = ToolGenerator()


class DomInfo1(BaseModel):
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


class InterpreGenerator(object):

    def __init__(self, llm_provider, llm_model):
        self.llm_service = LLMServiceFactory.create_llm_service(
            llm_provider=llm_provider,
            llm_model=llm_model,
            system_message=INTERPRE_SYSTEM_PROMPT,
            other_parameters={'temperature': 0.0})

        self.message_generator = MessageGenerator(llm_service=self.llm_service,
                                                  debug_mode=DEBUG_MODE)

    def extract_mathrm_operators(self, text: str) -> list[str]:
        """
        使用正则表达式从包含 LaTeX `\mathrm{}` 标记的字符串中提取所有算子名称。

        Args:
            text: 包含 LaTeX 标记的输入字符串。

        Returns:
            一个包含所有提取出的算子名称的列表。
        """
        # 正则表达式，用于查找 \mathrm{...} 结构并捕获其中的内容
        # 捕获组 ([a-zA-Z0-9_]+) 会匹配由字母、数字和下划线组成的算子名
        regex = r"\\mathrm\{([a-zA-Z0-9_]+)\}"

        # re.findall 会返回所有捕获组匹配到的字符串列表
        operators = re.findall(regex, text)

        return operators

    async def generate(self, features: Optional[str], operators: Optional[str],
                       expression: Optional[str]) -> Dict[str, str]:
        content = await self.message_generator.generate_message_async(
            messages=[
                HumanMessage(content=INTERPRE_USER_PROMPT.format(
                    **{
                        'user_features': features,
                        'operators': operators,
                        'expression': expression
                    }))
            ],
            output_schema=DomInfo1)
        return content.model_dump()


class InterpreFactorTool(BaseTool):
    name: str = "factor_interpre_tool"
    description: str = "擅长对因子进行逻辑解释和提供背后理论依据"
    parameters: dict = {
        "type": "object",
        "expression": {
            "expresss": {
                "type": "string"
            },
            "category": {
                "type": "string"
            }
        },
        "required": ["expression", "category"]
    }

    _interpre_generator: Optional[InterpreGenerator] = None

    def _initialize(self):
        if self._interpre_generator is None:
            self._interpre_generator = InterpreGenerator(
                llm_model=os.environ['MODEL_NAME'],
                llm_provider=os.environ['MODEL_PROVIDER'])

    async def execute(self, expression: str, category: str) -> ToolResult:
        try:
            self._initialize()
            tools = tool_generator.expression_disassembly(
                category=category, expression=expression)
            expr = await self._interpre_generator.generate(
                expression=expression,
                features=tools['features'],
                operators=tools['expression'])
            return ToolResult(output={"expression": expr})
        except Exception as e:
            return ToolResult(error=f"表达式生成失败: {e}")
