from typing import Any, Dict
from dichaos.services.mcp.agent import MCPAgent
from dichaos.services.mcp.until import generate_tool_prompt, TOOL_RULE_PROMPT

from tools.factor.exps_tool import ExpsFactorTool

EXPS_AGENT_SYSTEM_PROMPT = """
你是一位专业的量化因子实现专家。你的任务是接收一段因子的自然语言描述，然后使用 `factor_exps_tool` 工具将其精确地转换为一个可计算的数学表达式。你的输出应该是简洁和准确的。

**工作流程**:
1.  **分析**: 理解用户的自然语言描述。
2.  **行动**: 调用 `factor_exps_tool` 工具生成数学表达式。
3.  **总结**: 在成功获取到工具返回的表达式后，**此时不要再调用任何工具。**。 将工具返回内容以json返回
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


class ExpsAgent(MCPAgent):
    name: str = "exps_agent"
    description: str = "一个将因子描述转换为数学表达式的专家"
    system_prompt: str = EXPS_AGENT_SYSTEM_PROMPT
    tool_server_config: Dict[str, Any] = {"transport": "http"}

    system_prompt_openai: str = EXPS_AGENT_SYSTEM_PROMPT
    system_prompt_ollama: str = EXPS_AGENT_SYSTEM_PROMPT#f"{EXPS_AGENT_SYSTEM_PROMPT}\n\n{TOOL_RULE_PROMPT.format(name)}\n\n{generate_tool_prompt([ExpsFactorTool()])}"