import asyncio, pdb, json
import sys, os, argparse
from dotenv import load_dotenv

load_dotenv()
from dichaos.services.llm.factory import LLMServiceFactory
from agents.chip_agent import ChipAnalysisAgent
from agents.factor import IdeaAgent
from agents.factor import ExpsAgent
from agents.factor import BuildAgent
from agents.factor import Build1Agent
from agents.factor import DSLAgent
from agents.factor import ForgeAgent

### 临时输入
test_input = {
    'money_flow_volume_in': '主动买入的总成交量',
    'money_flow_volume_out': '主动卖出的总成交量',
    'vwap_tick': '单笔成交均价',
    'vwap_total': '时间窗⼝总成交均价',
    'money_flow_net_volume_in': '净主动买⼊成交量',
    'money_flow_smart_money_out': '聪明钱卖出⾦额（⾼于均价的主动卖单）'
}


async def run_agent_workflow(agent, user_request: str, **kwargs):
    """通用的 Agent 工作流执行器。"""
    try:
        pdb.set_trace()
        await agent.connect_tool_server()
        final_report = await agent.run(request=user_request, **kwargs)
        pdb.set_trace()
        print(final_report)
    finally:
        if agent:
            await agent.cleanup()


async def main():
    llm_service = LLMServiceFactory.create_llm_service(
        llm_model=os.environ['MODEL_NAME'],
        llm_provider=os.environ['MODEL_PROVIDER'],
        system_message='')

    agent_map = {
        'chip': (ChipAnalysisAgent, "分析股票 600519..."),
        'idea': (IdeaAgent, "我观察到市场，发现量价共振因子，具备强预测性"),
        "exps":
        (ExpsAgent,
         "因子名称：量价共振增强因子。计算方法：首先，计算过去N个交易日内每日的成交量变化率（当日成交量 - 前一日成交量）/ 前一日成交量，以及每日的价格变化率（当日收盘价 - 前一日收盘价）/ 前一日收盘价。然后，将每日的成交量变化率和价格变化率相乘，得到每日的量价协同指标."
         ),
        "build":
        (BuildAgent, "我观察到市场，发现量价共振因子，具备强预测性. 包括了成交量,持仓量,主动买入总成交量,主动卖出总成交量,"),
        "build1": (Build1Agent, "从工具集中推演因子"),
        "dsl": (DSLAgent, "从工具集中推演因子"),
        "forege": (ForgeAgent, json.dumps(test_input))
    }
    agent_class, user_request = agent_map['build1']


    agent_kwargs = {"llm_service": llm_service, "generate_final_report": False}

    server_base_url = f"http://0.0.0.0:8001"
    pdb.set_trace()
    #temp_agent_for_config = agent_class(**agent_kwargs)
    #agent_base_config = temp_agent_for_config.tool_server_config.copy()

    #agent_base_config["base_url"] = server_base_url
    agent_kwargs["tool_server_config"] = {
        'transport': 'http',
        #'module_path': 'tools.main_server',
        'base_url': server_base_url
    }
    agent = await agent_class.create(**agent_kwargs)
    workflow_params = {}

    await run_agent_workflow(agent, user_request, **workflow_params)


if __name__ == "__main__":
    asyncio.run(main())
