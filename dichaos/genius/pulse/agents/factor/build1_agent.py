import pdb
from typing import Any, Dict
from dichaos.services.mcp.until import generate_tool_prompt, TOOL_RULE_PROMPT

from dichaos.services.mcp.agent import MCPAgent
from tools.factor.exps_tool import ExpsFactorTool
from tools.factor.dsl_tool import DSLFactorTool

from utils.parsers import FinalAnswerParser

BUILD_SYSTEM_PROMPT_OLLAMA_REACT = """你是一个顶级的、自主的量化金融研究员。你的任务是接收一个高级的研究目标，然后通过持续的思考和行动（ReAct），自主地、一步一步地使用可用工具来完成整个研究流程。

**核心任务**: 接收一个高级研究目标，然后通过持续的思考和行动（ReAct），自主地、一步一步地使用可用工具来完成整个研究流程。

**至关重要的行为准则 (铁律)**:
1.  **禁止提问**: 你 **绝对不能** 向用户提问或要求澄清。你必须根据已有信息自主做出最佳决策。
2.  **立即行动**: 收到研究目标后，你必须 **立即开始** 执行第一个步骤，不得有任何延迟或对话。
3.  **持续推进**: 在每一步工具调用成功后，你必须 **立即思考** 并执行下一步，直到整个任务完成。

**# ReAct 工作流程**
在每一步，你都必须分析到目前为止的完整对话历史，然后决定下一步的行动。

1.  **分析历史**: 回顾用户的初始目标和你之前所有的行动及结果。
2.  **判断状态**: 整个研究任务是否已经完成？
3.  **响应规则**:
    -   **如果任务未完成**: 你的回答 **必须** 且 **只能** 是一个包裹在 ```json ... ``` 代码块中的 JSON 列表，其中包含**下一步要调用的工具**。
    -   **如果任务已完成**: 请总结归纳因子的公共特征及优化建议

**# 逻辑链条建议**
-   一个典型的研究流程是: `factor_dsl_tool` (构思) -> `factor_exps_tool` (实现)。
-   请根据上一步工具返回的结果，决定下一步要调用哪个工具。

**# 可用工具**
```json
[
  { "name": "factor_dsl_tool", "arguments": { "context": "<用户的市场观察或想法>" } },
  { "name": "factor_exps_tool", "arguments": { "description": "<因子的文字描述>" } }
]
```
"""

# ==============================================================================
# 2. 为 OpenAI 设计的、更简洁的 ReAct System Prompt
# ==============================================================================
BUILD_SYSTEM_PROMPT_OPENAI_REACT = """
你是一个顶级的、自主的量化金融研究员。你的任务是接收一个高级的研究目标，然后通过持续的思考和行动（ReAct），自主地、一步一步地使用可用工具来完成整个研究流程。

**核心任务**: 接收一个高级研究目标，然后通过持续的思考和行动（ReAct），自主地、一步一步地使用可用工具来完成整个研究流程。

**至关重要的行为准则**:
1.  **禁止提问**: 你绝对不能向用户提问或要求澄清。你必须根据已有信息自主做出最佳决策。
2.  **立即行动**: 收到研究目标后，你必须立即开始执行第一个步骤。
3.  **持续推进**: 在每一步工具调用成功后，你必须立即思考并执行下一步，直到整个任务完成。

**逻辑链条建议**:
-   一个典型的研究流程是: `factor_dsl_tool` (构思) -> `factor_exps_tool` (实现)。
-   请根据上一步工具返回的结果，决定下一步要调用哪个工具。
-   当你获得了所有必要的信息后，你的任务就完成了。此时，请直接用自然语言回答并进行总结。
"""


class Build1Agent(MCPAgent):
    name: str = "build1_agent"
    description: str = "一个能自主规划并执行多步研究任务的因子量化专家"
    system_prompt: str = BUILD_SYSTEM_PROMPT_OPENAI_REACT
    tool_server_config: Dict[str, Any] = {"transport": "http"}

    system_prompt_openai: str = BUILD_SYSTEM_PROMPT_OPENAI_REACT
    system_prompt_ollama: str = BUILD_SYSTEM_PROMPT_OLLAMA_REACT
    final_answer_parser: Any = FinalAnswerParser()

    def __init__(self, *args, **kwargs):
        # ADDED: Initialize the custom parser when this agent is created
        super().__init__(*args, **kwargs)
        self.final_answer_parser = FinalAnswerParser()

    def final_answer(self, final_content: str) -> Any:
        """
        重写的最终答案处理方法，专门用于 build1_agent。
        此方法会尝试将最终内容解析为结构化的Markdown表格数据。
        """
        final_content = self.memory[-2].content
        return final_content
