import pdb
from typing import Any, Dict
from dichaos.services.mcp.agent import MCPAgent
from dichaos.services.mcp.until import generate_tool_prompt, TOOL_RULE_PROMPT

from tools.factor.forge_tool import ForgeFactorTool

FORAGE_AGENT_SYSTEM_PROMPT = """
你是一位顶级的**量化因子发明家 (Factor Inventor)**。你的任务是使用 `factor_forge_tool` 工具来构思新的阿尔法因子。获取到工具返回的因子和设计描述，，然后以清晰的报告形式呈现给用户。
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
  "dsl":{{
    "hypothesis": "<在这里填写你基于工具箱构思出的假说>",
    "description": "<在这里填写因子设计的简要描述>"
}}
}}

"""


class ForgeAgent(MCPAgent):
    name: str = "forget_agent"
    description: str = "一个专业的因子推演专家。"
    system_prompt: str = FORAGE_AGENT_SYSTEM_PROMPT
    tool_server_config: Dict[str, Any] = {"transport": "http"}

    system_prompt_openai: str = FORAGE_AGENT_SYSTEM_PROMPT
    system_prompt_ollama: str = f"{FORAGE_AGENT_SYSTEM_PROMPT}\n\n{TOOL_RULE_PROMPT.format('factor_forge_tool')}\n\n{generate_tool_prompt([ForgeFactorTool()])}"
    
    def final_answer(self, final_content: str) -> Any:
      """
        重写的最终答案处理方法，专门用于 build1_agent。
        此方法会尝试将最终内容解析为结构化的Markdown表格数据。
      """
      pdb.set_trace()
      final_content = self.memory[-2].content
      return final_content
