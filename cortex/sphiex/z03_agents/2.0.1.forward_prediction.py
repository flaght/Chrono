import pdb, itertools, os, toml, asyncio, math, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from features.text.mode import TraderPredictionResult
from lib.inkits.utils import build_dynamic_schema, get_fewshot_template2
from lib.agt001 import create_agent
from lib.pdb001 import PromptDataBuilder
from kdutils.macro import base_path


## 统一文件命名 路径+ 文件名
def create_file(save_path, trade_date, ticker):
    file_path = Path(save_path) / f"{trade_date}_{ticker}.json"
    return file_path


def save_prediction_snapshot(trade_date, ticker, name, holding_period,
                             input_data_dict, prediction_output, save_path):
    snapshot = {
        "sample_id": f"SMP-{ticker}-{trade_date}",
        "trade_date": trade_date,
        "ticker": ticker,
        "name": name,
        "holding_period": holding_period,
        "created_at": datetime.now().isoformat(),
        # 原始输入数据快照 (Point-in-Time 冻结)
        "input_context": {
            "regime_features_table": input_data_dict.get("regime_str"),
            "predictive_signals_table": input_data_dict.get("predictive_str"),
            "textual_events_timeline": input_data_dict.get("textual_str"),
        },
        # Trader 预测结果
        "trader_prediction": prediction_output,
        # 状态标记：等待真实收益结算
        "status": "PENDING_SETTLEMENT",
        "forward_return": None,
        "reviewer_result": None
    }
    Path(save_path).mkdir(parents=True, exist_ok=True)
    file_path = create_file(save_path, trade_date, ticker)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    return str(file_path)


async def generate_with_semaphore(agent_instance,
                                  trade_time,
                                  semaphore,
                                  human_message,
                                  params_dict,
                                  schema_cls,
                                  key_name='events'):
    async with semaphore:
        print(f"[{agent_instance.name}] 获取到并发许可，开始极速发散推演...")
        try:
            result = await agent_instance.agenerate_message(
                human_message=human_message,
                params=params_dict,
                response_schema=schema_cls)
            output = result.model_dump()
            output['name'] = agent_instance.name
            output['trade_time'] = trade_time
            return {"output": output, "status": 0, "input": params_dict}
        except Exception as e:
            print(f"❌ [{agent_instance.name}] 生成期间发生错误: {e}")
            return {"output": "", "status": -1}


def load_data(method):
    predict_data = pd.read_feather(
        os.path.join("records", "normal", str(method), "predict_data.feather"))
    regime_data = pd.read_feather(
        os.path.join("records", "normal", str(method), "regime_data.feather"))
    textuals_data = pd.read_feather(
        os.path.join("records", "normal", str(method),
                     "textuals_data.feather"))
    # returns_data = pd.read_feather(
    #     os.path.join("records", "basic", str(method), "returns_data.feather"))
    predict_data['trade_date'] = pd.to_datetime(predict_data['trade_date'])
    regime_data['trade_date'] = pd.to_datetime(regime_data['trade_date'])
    textuals_data['trade_date'] = pd.to_datetime(textuals_data['trade_date'])
    return predict_data, regime_data, textuals_data


async def create_predict_agent():
    llm_name = 'deepseek_4001'  ## 指定大模型 包括地址 参数 都存储在对应字典
    vector_name = 'embedding_10002'  ##  指定嵌入模型  包括地址 参数 都存储在对应字典
    persona_name = 'quant_fusion_trader_10001'
    thoughts_name = 'fusion_trader_user_100001'
    agent_name = "reflection1"
    agent_title = "交易反思者"
    vector_provider = 'fassis'
    agent, thoughts1 = await create_agent(llm_name=llm_name,
                                          vector_name=vector_name,
                                          persona_name=persona_name,
                                          thoughts_name=thoughts_name,
                                          agent_name=agent_name,
                                          agent_title=agent_title,
                                          category="diver")
    return agent, thoughts1, thoughts_name


## 训练agent经验 agent, 先批量预测任务，再批量反思任务
async def predict(method, period, lookback=3, is_refresh=False):

    async def run_task(tasks):
        batch_results = await asyncio.gather(*tasks)
        for result in batch_results:
            save_prediction_snapshot(trade_date=result['output']['trade_time'],
                                     ticker=ticker,
                                     name=name,
                                     holding_period=holding_period,
                                     input_data_dict=result['input'],
                                     prediction_output=result['output'],
                                     save_path=save_path)

    predict_data, regime_data, textuals_data = await asyncio.to_thread(
        load_data, method=method)

    predict_agent, predict_thoughts, predict_thoughts_name = await create_predict_agent(
    )
    dates = set(predict_data['trade_date']).intersection(
        regime_data['trade_date'], textuals_data['trade_date'])
    dates = [d.strftime('%Y-%m-%d') for d in dates]
    dates.sort()
    #dates = dates[0:lookback + 2]
    ticker = "000852"
    name = '中证1000指数 (000852.SH / IM)'
    holding_period = "T+1开盘 ~ T+{}开盘".format(period + 1)
    semaphore = asyncio.Semaphore(4)
    tasks = []
    save_path = os.path.join(base_path, "prod", str(method), str(period))
    os.makedirs(save_path, exist_ok=True)
    for index, date in enumerate(dates):
        if index < lookback:
            continue
        end_date = date
        start_date = dates[index - lookback]

        filename = create_file(save_path=save_path,
                               trade_date=end_date,
                               ticker=ticker)
        if (os.path.exists(filename) and not is_refresh):
            continue
        
        pdata = predict_data[(predict_data['trade_date'] >= start_date)
                             & (predict_data['trade_date'] <= end_date)]
        rdata = regime_data[(regime_data['trade_date'] >= start_date)
                            & (regime_data['trade_date'] <= end_date)]
        tdata = textuals_data[(textuals_data['trade_date'] >= start_date)
                              & (textuals_data['trade_date'] <= end_date)]
        
        predictive_str = PromptDataBuilder.build_predictive_signals(pdata)
        regime_str = PromptDataBuilder.build_regime_features(rdata)
        textual_str = PromptDataBuilder.build_textual_events(tdata)
        memories_str = PromptDataBuilder.build_retrieved_memories([])

        json_structure = get_fewshot_template2(TraderPredictionResult)

        params = {
            "ticker": ticker,
            "name": name,
            "holding_period": holding_period,
            "regime_features_time_series": regime_str,
            "predictive_signals_time_series": predictive_str,
            "textual_events_data": textual_str,
            "retrieved_memories_block": memories_str,
            "json_structure": json_structure
        }
        thought = predict_thoughts[predict_thoughts_name]
        thougths_cot = thought["cot"]
        prompt = f"""
            {thougths_cot}
            \n\n输出格式必须为:\n
            {{json_structure}}
            """
        tasks.append(
            generate_with_semaphore(trade_time=end_date,
                                    agent_instance=predict_agent,
                                    semaphore=semaphore,
                                    human_message=prompt,
                                    params_dict=params,
                                    schema_cls=TraderPredictionResult))
        if len(tasks) >= 1:
            await run_task(tasks)
            tasks = []
    if len(tasks) > 0:
        await run_task(tasks)


if __name__ == '__main__':
    method = 'train0'
    asyncio.run(predict(method=method, period=3, lookback=3))
