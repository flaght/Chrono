from typing import Any, Dict
from dichaos.services.mcp.agent import MCPAgent
from dichaos.services.mcp.until import generate_tool_prompt, TOOL_RULE_PROMPT

from tools.factor.dsl_tool import DSLFactorTool

DSL_AGENT_SYSTEM_PROMPT = """
你是一位顶级的量化研究员。使用 `factor_dsl_tool` 工具来构思一个新的阿尔法因子。获取到工具返回的因子设想后，你需要对其进行评估和润色，然后以清晰的报告形式呈现给用户。
**工作流程**:
1.  **行动**: 调用 `factor_idea_tool` 工具生成因子设想
3.  **总结**: 在成功获取到工具返回的因子设想后，**此时不要再调用任何工具。**。 将工具返回内容以json返回
必须严格按照如下格式:
    {{
    "hypothesis": "str",
    "description": "str"
    }}
"""


class DSLAgent(MCPAgent):
    name: str = "dsl_agent"
    description: str = "一个专业的因子设计专家。"
    system_prompt: str = DSL_AGENT_SYSTEM_PROMPT
    tool_server_config: Dict[str, Any] = {"transport": "http"}

    system_prompt_openai: str = DSL_AGENT_SYSTEM_PROMPT
    system_prompt_ollama: str = DSL_AGENT_SYSTEM_PROMPT#f"{DSL_AGENT_SYSTEM_PROMPT}\n\n{TOOL_RULE_PROMPT.format(name)}\n\n{generate_tool_prompt([DSLFactorTool()])}"
