from typing import Any, Dict
from langchain_core.messages import SystemMessage
from dichaos.services.mcp.agent import MCPAgent
from dichaos.services.mcp.until import generate_tool_prompt

from tools.chip_tool import ChipAnalysisTool

CHIP_ANALYSIS_SYSTEM_PROMPT = """你是一位专业的筹码分析师。你的任务是使用名为 `chip_analysis_tool` 的工具来获取股票的筹码数据。拿到数据后，你需要根据返回的筹码核心指标，为用户生成一份简洁、专业、易于理解的分析报告。报告应清晰地解读平均成本、获利比例和90%成本集中度这三个指标的含义，并给出一个综合的、非投资建议的结论。"""


class ChipAnalysisAgent(MCPAgent):
    name: str = "chip_analysis_agent"
    description: str = "一个专业的筹码分析专家。"
    tool_server_config: Dict[str, Any] = {
        "transport": "http",
        #"module_path": "tools.main_server"
    }

    system_prompt_openai: str = CHIP_ANALYSIS_SYSTEM_PROMPT
    system_prompt_ollama: str = f"{CHIP_ANALYSIS_SYSTEM_PROMPT}\n\n{generate_tool_prompt([ChipAnalysisTool()])}"
