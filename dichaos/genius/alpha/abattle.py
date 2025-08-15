import os, pdb, asyncio, time
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar
from dichaos.kdutils import kd_logger
from dichaos.battle.environment import AsyncBattleEnvironment
from dichaos.battle.state import BattleState
from agent.agents import Agents
from agent import IndicatorPredict, CloutoPredict, MoneyFlowPredict
from agent.decision.agent import DecisionAgent
from kdutils.report import ReportGenerator


async def run_concurrent_predictions(model_date: str, end_date: str,
                                     symbol: str):
    predictors = [
        CloutoPredict(date=model_date,
                      memory_path=os.path.join("records"),
                      symbol=symbol),
        IndicatorPredict(date=model_date,
                         memory_path=os.path.join("records"),
                         symbol=symbol),
        MoneyFlowPredict(date=model_date,
                         memory_path=os.path.join("records"),
                         symbol=symbol)
    ]

    end_date = advanceDateByCalendar('china.sse', end_date,
                                     '-{0}b'.format(1)).strftime('%Y-%m-%d')

    ### 串行数据准备
    for p in predictors:
        p.prepare_data(begin_date=end_date, end_date=end_date)

    ### 并发
    tasks = [
        p.agenerate_prediction(date=end_date,
                               predict_data=p.create_data(date=end_date))
        for p in predictors
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理并收集结果
    analysis_reports = {}  # key 是 agent.name, value 是分析报告字符串
    reason_reports = {}
    predictor_agents = []

    for pred, result in zip(predictors, results):
        agent_name = pred.agent.name
        predictor_agents.append(pred.agent)  # 收集 Agent 实例

        summary = getattr(result, 'summary', str(result.summary))

        reason = getattr(result, 'reasoning', str(result.reasoning))

        analysis_details = getattr(result, 'analysis_details',
                                   str(result.analysis_details))

        analysis_reports[agent_name] = "{0} {1}".format(
            analysis_details, summary)

        reason_reports[agent_name] = reason

        kd_logger.info(f"📊 来自 {pred.agent.name} 的分析")

    return analysis_reports, reason_reports, predictor_agents


async def run_agent_debate(symbol: str, initial_reports_map: Dict[str, str],
                           predictor_agents: List[Agents]):
    kd_logger.info("⚔️ 阶段二: 复用 Agent 进行博弈与决策 ⚔️")
    environment = AsyncBattleEnvironment(debate_rounds=5)

    for agent_instance in predictor_agents:
        agent_name = agent_instance.name
        role = agent_instance.desc()
        initial_analysis_for_agent = initial_reports_map.get(
            agent_name, "我没有生成初始分析。")
        environment.register_agent(agent=agent_instance,
                                   role_description=role,
                                   initial_analysis=initial_analysis_for_agent)

    # 将所有报告汇总，作为辩论的全局上下文
    report_str = "\n".join([
        f"- **{report_name}**: {report_text}"
        for report_name, report_text in initial_reports_map.items()
    ])
    research_report = {
        "symbol": symbol,
        "preliminary_analysis_summary": report_str
    }

    # 启动辩论
    final_results = await environment.run(research_report)
    return final_results


async def run_final_decision(decision_agent: DecisionAgent,
                             battle_state: BattleState, symbol: str,
                             date: str):
    pdb.set_trace()
    kd_logger.info("🎓 阶段三: 最终决策合成 🎓")
    # 1. 格式化辩论历史为单个字符串
    debate_transcript = []
    current_round = 0
    for event in battle_state['debate_history']:
        #if event["round"] != current_round and event["round"] == 3:
        if event["round"] == 3 or event["round"] == 4 or event["round"] == 5:
            current_round = event["round"]
            debate_transcript.append(f"\n--- 第 {current_round} 轮 ---")
            debate_transcript.append(f'{event["speaker"]}: {event["content"]}')

    full_transcript = "\n".join(debate_transcript)
    kd_logger.info("📜 辩论记录摘要: {0}".format(full_transcript))

    final_prediction = await decision_agent.agenerate_prediction(
        debate_transcript=full_transcript, symbol=symbol, date=date)
    return final_prediction


async def main_workflow():
    symbol = '601519'
    model_date = '2025-01-27'
    end_date = '2025-02-14'

    ##构建决策agent
    decision_agent = DecisionAgent.from_config(
        path=os.path.join('agent', DecisionAgent.name))

    # --- 阶段一：并发预测 ---
    initial_analysis_map, reason_reports, predictor_agents = await run_concurrent_predictions(
        model_date=model_date, end_date=end_date, symbol=symbol)

    # --- 阶段二：多智能体辩论 ---
    final_battle_state = await run_agent_debate(
        symbol=symbol,
        initial_reports_map=initial_analysis_map,
        predictor_agents=predictor_agents)

    # --- ### 新增：阶段三：最终决策 ---
    final_prediction = await run_final_decision(
        decision_agent=decision_agent,
        battle_state=final_battle_state,
        symbol=symbol,
        date=end_date)

    report_data = {
        "date": end_date,
        "symbol": symbol,
        "final_prediction": final_prediction.model_dump(),
        "battle_results": final_battle_state,
        "reason_reports": reason_reports
    }
    pdb.set_trace()
    report = ReportGenerator(output_dir=os.path.join("records", "report",
                                                     "html", end_date),
                             template_name=os.path.join(
                                 "resource", "report_template.html"))
    report.run(report_data=report_data)


if __name__ == "__main__":
    #agent_predict1()
    asyncio.run(main_workflow())
