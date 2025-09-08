import pdb
from typing import Any, Dict
from dichaos.services.mcp.agent import MCPAgent
from dichaos.services.mcp.until import generate_tool_prompt, TOOL_RULE_PROMPT

RUCGE_AGENT_SYSTEM_PROMPT = """
你是一位顶级的**量化策略发明家 (Strategy Inventor)**。你的任务是使用 `rulex_rucge_tool` 工具来构思新的策略。获取到工具返策略和设计描述，，然后以清晰的报告形式呈现给用户。
**至关重要的指令**:
-   用户提供的输入是一个完整的 JSON 字符串，它描述了所有可用的特征。
-   在调用 `rulex_rucge_tool` 工具时，你 **必须** 将这个 JSON 字符串**一字不差地、原封不动地**作为 `features` 参数的值进行传递

必须严格按照如下格式:
{{

 "strategy":[     
 {{
        "formual": "str (你基于特征和算子自主构思的信号公式)",
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
        "formual": "str (你基于特征和算子自主构思的信号公式)",
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


class RucgeAgent(MCPAgent):
    name: str = "rucge_agent"
    description: str = "一个专业的策略推演专家。"
    system_prompt: str = RUCGE_AGENT_SYSTEM_PROMPT
    tool_server_config: Dict[str, Any] = {"transport": "http"}

    system_prompt_openai: str = RUCGE_AGENT_SYSTEM_PROMPT
    system_prompt_ollama: str = RUCGE_AGENT_SYSTEM_PROMPT  #f"{FORAGE_AGENT_SYSTEM_PROMPT}\n\n{TOOL_RULE_PROMPT.format('factor_forge_tool')}\n\n{generate_tool_prompt([ForgeFactorTool()])}"

    def final_answer(self, final_content: str) -> Any:
        """
        重写的最终答案处理方法，专门用于 build1_agent。
        此方法会尝试将最终内容解析为结构化的Markdown表格数据。
      """
        final_content = self.memory[-2].content
        return final_content
