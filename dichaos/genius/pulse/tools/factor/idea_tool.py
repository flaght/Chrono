import pdb, os
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple
from langchain_core.messages import HumanMessage
from dichaos.services.react.tool import BaseTool, ToolResult
from dichaos.services.llm import LLMServiceFactory, MessageGenerator
from dichaos.env import DEBUG_MODE

IDEA_SYSTEM_PROMPT = """
    你是一位顶级的**量化策略师**与**因子研究专家**，一个致力于探索市场深层规律、寻找新型Alpha来源的AI大脑。你知识渊博、富有创造力，并始终坚持逻辑的严谨性。

    你的核心任务是将深刻的市场洞察，转化为一个结构清晰、逻辑严谨、具备创新性的Alpha因子提案。
    **你的提案必须遵循以下三大原则：**
        1.  **逻辑自洽 (Logically Consistent):** 市场假说与因子描述之间必须存在清晰的因果关系或逻辑链条。
        2.  **可实现性 (Feasible):** 因子描述应基于合理可得的数据（无论是传统的价量、财务数据，还是另类数据），并具备清晰的可操作性。
        3.  **创新性 (Innovative):** 鼓励提出能捕捉到特定市场异象（Anomaly）、投资者行为偏差或结构性机会的新颖想法，而非重复已知的经典因子。
"""
IDEA_USER_PROMPT = """
    作为一名经验丰富的量化研究员，你被要求评估以下市场观察并提出一个可行的Alpha因子。
    **市场观察**:
    > "{context}"

    **你的任务包含三个部分:**

    1.  **核心假说构建 (Hypothesis Formulation)**:
        *   从上述观察中提炼出一个核心的、可盈利的市场假说。这个假说必须清晰地解释其背后的经济学或行为金融学原理。

    2.  **因子量化设计 (Factor Specification)**:
        *   设计一个具体的因子来捕捉这个假说。描述其计算方法, 重视逻辑性和真实性，不允许直接生成因子表达式。注重在设计描述。不要加入任何与因子设计无关的内容。

    3.  **潜在风险与失效情景 (Risk & Failure Scenarios)**:
        *   **（思考过程，无需在最终JSON中输出）** *请在内心思考：这个假说在什么情况下可能会失效？例如，如果市场结构变化、所有人都知道了这个模式，或者这个观察只是短期噪音怎么办？* 将这些思考融入你的因子设计中，使其更具鲁棒性。

    **最终输出**
    请将你的**假说**和**因子设计**严格按照以下JSON格式返回。你的因子描述应体现出对潜在风险的考量。
    {{
    "hypothesis": "<你的假-说>",
    "description": "<你的因子描述>"
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
            system_message=IDEA_SYSTEM_PROMPT,
            other_parameters={'temperature': 1.0})

        self.message_generator = MessageGenerator(llm_service=self.llm_service,
                                                  debug_mode=DEBUG_MODE)

    async def generate(self, context: Optional[str] = None) -> Dict[str, str]:
        content = await self.message_generator.generate_message_async(
            messages=[
                HumanMessage(content=IDEA_USER_PROMPT.format(
                    **{'context': context}))
            ],
            output_schema=DomInfo)
        return content.model_dump()


class IdeaFactorTool(BaseTool):
    name: str = "factor_idea_tool"
    description: str = "基于市场观察，提出一个新的阿尔法因子的设想。"
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
            output = await self._idea_generator.generate(context=context)
            if not output.get("hypothesis"):
                return ToolResult(error="LLM未能生成有效的因子设想。")
            return ToolResult(output=output)
        except Exception as e:
            return ToolResult(error=f"因子设想失败: {e}")
