import os, pdb, asyncio, time
import pandas as pd
from typing import Dict, List
from joblib import Parallel, delayed
from urllib.parse import urlparse
from pymongo import InsertOne, DeleteOne

from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar
from dichaos.battle.environment import AsyncBattleEnvironment
from dichaos.battle.state import BattleState
from agent.agents import Agents
from agent import IndicatorPredict, CloutoPredict, MoneyFlowPredict, ChipPredict, HotMoneyPredict
from agent.decision.agent import DecisionAgent
from kdutils.report import ReportGenerator
from kdutils.mongo import MongoLoader

mongo_client = MongoLoader(connection_string=os.environ['MG_URL'],
                           db_name=urlparse(
                               os.environ['MG_URL']).path.lstrip('/'),
                           collection_name='chat_history1')


async def predict_with_semaphore(predictor, semaphore: asyncio.Semaphore,
                                 date: str):
    """
    一个异步包装函数，它在使用 predictor 进行预测之前，会先从 semaphore 获取许可。
    
    Args:
        predictor: 一个 Predictor 类的实例 (例如 CloutoPredict)。
        semaphore (asyncio.Semaphore): 用于控制并发的信号量。
        date (str): 预测日期。

    Returns:
        The result of the prediction or the exception if it fails.
    """
    # 3. 在包装协程内获取 Semaphore
    # async with 语句确保了即使在预测过程中发生异常，信号量也总能被正确释放。
    async with semaphore:
        # 当协程执行到这里时，它已经成功获取了一个“令牌”。
        # 如果令牌已满，它会在上一行异步地等待。
        print(f"[{predictor.agent.name}] 获取到信号量许可，开始执行预测...")

        try:
            # 执行实际的预测调用
            result = await predictor.agenerate_prediction(
                date=date, predict_data=predictor.create_data(date=date))
            return result
        except Exception as e:
            # 捕获并返回异常，这样 gather 就不会中断
            print(f"[{predictor.agent.name}] 预测时发生错误: {e}")
            return e


async def run_concurrent_predictions2(model_date: str, end_date: str,
                                      symbol: str):
    MAX_CONCURRENT_REQUESTS = 2
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    print(f"信号量已创建，最大并发预测数为: {MAX_CONCURRENT_REQUESTS}")

    predictors = [
        CloutoPredict(date=model_date,
                      memory_path=os.path.join("records"),
                      symbol=symbol),
        IndicatorPredict(date=model_date,
                         memory_path=os.path.join("records"),
                         symbol=symbol)
    ]

    end_date = advanceDateByCalendar('china.sse', end_date,
                                     '-{0}b'.format(1)).strftime('%Y-%m-%d')

    for p in predictors:
        p.prepare_data(begin_date=end_date, end_date=end_date)

    # 创建一个任务列表，这次调用的是我们的包装函数 predict_with_semaphore
    tasks = [
        predict_with_semaphore(p, semaphore, end_date) for p in predictors
    ]
    print("已创建所有并发任务，准备使用 asyncio.gather 运行...")
    results = await asyncio.gather(*tasks)
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


async def run_concurrent_predictions1(model_date: str, end_date: str,
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
                         symbol=symbol),
        ChipPredict(date=model_date,
                    memory_path=os.path.join("records"),
                    symbol=symbol),
        HotMoneyPredict(date=model_date,
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


async def run_agent_debate(symbol: str,
                           initial_reports_map: Dict[str, str],
                           predictor_agents: List[Agents],
                           debate_rounds=4):
    print("⚔️ 阶段二: 复用 Agent 进行博弈与决策 ⚔️")
    environment = AsyncBattleEnvironment(debate_rounds=debate_rounds)

    for agent_instance in predictor_agents:
        agent_name = agent_instance.name
        role = agent_instance.desc()
        initial_analysis_for_agent = initial_reports_map.get(
            agent_name, "我没有生成初始分析。")
        pdb.set_trace()
        environment.register_agent(llm_provider=agent_instance.llm_provider,
                                   agent=agent_instance,
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
                             battle_state: BattleState, symbol: str, date: str,
                             debate_rounds: int):
    print("🎓 阶段三: 最终决策合成 🎓")
    # 1. 格式化辩论历史为单个字符串
    debate_transcript = []
    current_round = 0
    for event in battle_state['debate_history']:
        #if event["round"] != current_round and event["round"] == 3:
        if event["round"] in [
                debate_rounds, debate_rounds - 1, debate_rounds - 2
        ]:
            current_round = event["round"]
            debate_transcript.append(f"\n--- 第 {current_round} 轮 ---")
            debate_transcript.append(f'{event["speaker"]}: {event["content"]}')

    full_transcript = "\n".join(debate_transcript)
    print("📜 辩论记录摘要: {0}".format(full_transcript))

    final_prediction = await decision_agent.agenerate_prediction(
        debate_transcript=full_transcript, symbol=symbol, date=date)
    return final_prediction


# 将单个股票的完整处理流程封装成一个独立的异步函数
async def process_single_symbol_workflow2(symbol: str, model_date: str,
                                          end_date: str, debate_rounds: int):
    """
    处理单个股票的完整三阶段工作流。
    """
    #try:
    print(f"🚀 开始处理股票: {symbol}...")
    start_time = time.time()

    # --- 提前初始化决策 Agent ---
    # 注意：如果 DecisionAgent 的加载是 I/O 密集型，也可以在协程中做
    decision_agent = DecisionAgent.from_config(
        path=os.path.join('agent', DecisionAgent.name))

    # --- 阶段一：并发预测 ---
    initial_analysis_map, reason_reports, predictor_agents = await run_concurrent_predictions2(
        model_date=model_date, end_date=end_date, symbol=symbol)

    # --- 阶段二：多智能体辩论 ---
    final_battle_state = await run_agent_debate(
        symbol=symbol,
        initial_reports_map=initial_analysis_map,
        predictor_agents=predictor_agents,
        debate_rounds=debate_rounds)

    # --- 阶段三：最终决策 ---
    final_prediction = await run_final_decision(
        decision_agent=decision_agent,
        battle_state=final_battle_state,
        symbol=symbol,
        date=end_date,
        debate_rounds=debate_rounds)

    # --- 阶段四：生成报告 ---
    report_data = {
        "date": end_date,
        "symbol": symbol,
        "final_prediction": final_prediction.model_dump(),
        "battle_results": final_battle_state,
        "reason_reports": reason_reports
    }
    pdb.set_trace()
    base_path = "/workspace/worker/temp/nginx/opts/dichaos/stock"
    report = ReportGenerator(output_dir=os.path.join(base_path, str(end_date),
                                                     "report", "html"),
                             template_name=os.path.join(
                                 "resource", "report_template.html"))
    # 假设 report.run() 是同步的。如果是异步的，需要 await
    report.run(report_data=report_data)

    elapsed_time = time.time() - start_time
    print(f"✅ 成功处理股票: {symbol}，耗时: {elapsed_time:.2f} 秒")
    return {
        "symbol": symbol,
        "status": "success",
        "duration": elapsed_time,
        "report_data": report_data
    }
    '''
    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals(
        ) else 0
        print(f"❌ 处理股票: {symbol} 时发生错误: {e}")
        return {
            "symbol": symbol,
            "status": "failed",
            "error": str(e),
            "duration": elapsed_time
        }
    '''


# 将单个股票的完整处理流程封装成一个独立的异步函数
async def process_single_symbol_workflow1(symbol: str, model_date: str,
                                          end_date: str, debate_rounds: int):
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
        initial_analysis_map, reason_reports, predictor_agents = await run_concurrent_predictions1(
            model_date=model_date, end_date=end_date, symbol=symbol)

        # --- 阶段二：多智能体辩论 ---
        final_battle_state = await run_agent_debate(
            symbol=symbol,
            initial_reports_map=initial_analysis_map,
            predictor_agents=predictor_agents,
            debate_rounds=debate_rounds)

        # --- 阶段三：最终决策 ---
        final_prediction = await run_final_decision(
            decision_agent=decision_agent,
            battle_state=final_battle_state,
            symbol=symbol,
            date=end_date,
            debate_rounds=debate_rounds)

        # --- 阶段四：生成报告 ---
        report_data = {
            "date": end_date,
            "symbol": symbol,
            "final_prediction": final_prediction.model_dump(),
            "battle_results": final_battle_state,
            "reason_reports": reason_reports
        }
        base_path = "/workspace/worker/temp/nginx/opts/dichaos/stock"
        report = ReportGenerator(
            output_dir=os.path.join(base_path, str(end_date), "report",
                                    "html"),
            template_name=os.path.join("resource", "report_template.html"))
        # 假设 report.run() 是同步的。如果是异步的，需要 await
        report.run(report_data=report_data)

        elapsed_time = time.time() - start_time
        print(f"✅ 成功处理股票: {symbol}，耗时: {elapsed_time:.2f} 秒")
        return {
            "symbol": symbol,
            "status": "success",
            "duration": elapsed_time,
            "report_data": report_data
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
        process_single_symbol_workflow2(symbol, model_date, end_date, 2))


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


def update_data(results, table_name):
    delete_requests = [
        DeleteOne(message)
        for message in results[['symbol', 'trade_date']].to_dict(
            orient='records')
    ]

    insert_requests = [
        InsertOne(message) for message in results.to_dict(orient='records')
    ]

    requests = delete_requests + insert_requests

    mongo_client.bulk(requests=requests,
                      collection_name="{0}".format(table_name))


if __name__ == "__main__":
    symbols_to_process = [
        '001400', '605303', '002871', '000782', '603320', '605068', '000892',
        '601798', '002674', '002981', '002343', '002164', '600400', '603211',
        '002708', '000796', '001339', '002334', '002733', '600654', '603950',
        '600226', '000561'
    ][10:]
    symbols_to_process = ['601609']
    symbols_to_process = symbols_to_process

    end_date = '2025-08-20'  ## 预测的时间
    results = Parallel(n_jobs=1,
                       verbose=1)(delayed(run_workflow_entrypoint)(
                           symbols_to_process[i], '2025-01-27', end_date)
                                  for i in range(len(symbols_to_process)))
    pdb.set_trace()
    results = pd.DataFrame(results)
    results['trade_date'] = end_date
    update_data(results=results, table_name='genius_debate')
