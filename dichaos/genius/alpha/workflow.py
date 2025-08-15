import os, pdb, asyncio, time
from typing import Dict, List
from joblib import Parallel, delayed

from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar
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

        print(f"📊 来自 {pred.agent.name} 的分析")

    return analysis_reports, reason_reports, predictor_agents


async def run_agent_debate(symbol: str, initial_reports_map: Dict[str, str],
                           predictor_agents: List[Agents]):
    print("⚔️ 阶段二: 复用 Agent 进行博弈与决策 ⚔️")
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
    print("🎓 阶段三: 最终决策合成 🎓")
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
    print("📜 辩论记录摘要: {0}".format(full_transcript))

    final_prediction = await decision_agent.agenerate_prediction(
        debate_transcript=full_transcript, symbol=symbol, date=date)
    return final_prediction


# 将单个股票的完整处理流程封装成一个独立的异步函数
async def process_single_symbol_workflow(symbol: str, model_date: str,
                                         end_date: str):
    """
    处理单个股票的完整三阶段工作流。
    """
    try:
        print(f"🚀 开始处理股票: {symbol}...")
        start_time = time.time()

        # --- 提前初始化决策 Agent ---
        # 注意：如果 DecisionAgent 的加载是 I/O 密集型，也可以在协程中做
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

        # --- 阶段三：最终决策 ---
        final_prediction = await run_final_decision(
            decision_agent=decision_agent,
            battle_state=final_battle_state,
            symbol=symbol,
            date=end_date)

        # --- 阶段四：生成报告 ---
        report_data = {
            "date": end_date,
            "symbol": symbol,
            "final_prediction": final_prediction.model_dump(),
            "battle_results": final_battle_state,
            "reason_reports": reason_reports
        }

        report = ReportGenerator(
            output_dir=os.path.join("records", "report", "html"),
            template_name=os.path.join("resource", "report_template.html"))
        # 假设 report.run() 是同步的。如果是异步的，需要 await
        report.run(report_data=report_data)

        elapsed_time = time.time() - start_time
        print(f"✅ 成功处理股票: {symbol}，耗时: {elapsed_time:.2f} 秒")
        return {
            "symbol": symbol,
            "status": "success",
            "duration": elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals(
        ) else 0
        print(f"❌ 处理股票: {symbol} 时发生错误: {e}", exc_info=True)
        return {
            "symbol": symbol,
            "status": "failed",
            "error": str(e),
            "duration": elapsed_time
        }


def run_workflow_entrypoint(symbol, model_date, end_date):
    """
    这个同步函数是 ProcessPoolExecutor 调用的目标。
    它的作用是启动 asyncio 事件循环来运行我们的异步工作流。
    """
    #symbol, model_date, end_date = args_tuple
    return asyncio.run(
        process_single_symbol_workflow(symbol, model_date, end_date))


import numpy as np
from joblib import cpu_count


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=np.int32)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


if __name__ == "__main__":
    symbols_to_process = [
        '601519', '600519', '000001', '000002', '300750', '688981'
    ][:5]

    population = Parallel(n_jobs=4, verbose=1)(
        delayed(run_workflow_entrypoint)(symbols_to_process[i], '2025-01-27',
                                         '2025-02-14') for i in range(4))
