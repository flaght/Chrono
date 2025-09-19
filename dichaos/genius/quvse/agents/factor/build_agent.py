from typing import Any, Dict
from dichaos.services.mcp.agent import MCPAgent
from dichaos.services.mcp.until import generate_tool_prompt, TOOL_RULE_PROMPT

from tools.factor.exps_tool import ExpsFactorTool
from tools.factor.idea_tool import IdeaFactorTool

BUILD_SYSTEM_PROMPT_BASE = """你是一个顶级的、自主的量化金融研究员。你的任务是接收一个高级的研究目标，然后通过持续的思考和行动（ReAct），自主地、一步一步地使用可用工具来完成整个研究流程。

**# ReAct 工作流程**
在每一步，你都必须分析到目前为止的完整对话历史，然后决定下一步的行动。

1.  **分析历史**: 回顾用户的初始目标和你之前所有的行动及结果。
2.  **判断状态**: 整个研究任务是否已经完成？
    -   如果任务**未完成**，下一步最合理的行动是什么？
    -   如果任务**已完成**（例如，你已经成功构思、实现并评估了一个因子），下一步的行动就是总结报告。
{0}
**# 逻辑链条建议**
-   一个典型的研究流程是: `factor_idea_tool` -> `factor_exps_tool`。
-   请根据上一步工具返回的结果，决定下一步要调用哪个工具。


将工具返回内容以json返回
必须严格按照如下格式:
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
}}
"""

RESPONSE_RULE = """
3.  **响应规则**:
    -   **如果需要继续调用工具**: 你的回答 **必须** 且 **只能** 是一个包裹在 ```json ... ``` 代码块中的 JSON 列表，其中包含**下一步要调用的那一个工具**。
    -   **如果任务已完成**: 必须 **仅仅** 是一个包含了研究结果的 JSON 对象，**绝对不能** 包含任何 JSON 之外的文字
"""

BUILD_SYSTEM_PROMPT_OPENAI = BUILD_SYSTEM_PROMPT_BASE.format("")
BUILD_SYSTEM_PROMPT_OLLAMA = BUILD_SYSTEM_PROMPT_BASE.format(RESPONSE_RULE)

class BuildAgent(MCPAgent):
    name: str = "build_agent"
    description: str = "一个能自主规划并执行多步研究任务的因子量化专家"
    system_prompt: str = BUILD_SYSTEM_PROMPT_OPENAI
    tool_server_config: Dict[str, Any] = {"transport": "http"}

    system_prompt_openai: str = BUILD_SYSTEM_PROMPT_OPENAI
    system_prompt_ollama: str = BUILD_SYSTEM_PROMPT_OPENAI#f"{BUILD_SYSTEM_PROMPT_OLLAMA}\n\n{TOOL_RULE_PROMPT.format(name)}\n\n{generate_tool_prompt([IdeaFactorTool(),ExpsFactorTool()])}"
