import pdb, itertools, os, toml, asyncio, math, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro import base_path
from lib.inkits.utils import build_dynamic_schema, get_fewshot_template2
from lib.agt001 import create_agent
from lib.pdb001 import PromptDataBuilder
from features.text.mode import CaseMemoryReviewResult


def save_attribution_snapshot(trade_date, ticker, name, holding_period,
                              attribution_output, save_path):
    snapshot = {
        "sample_id": f"SMP-{ticker}-{trade_date}",
        "trade_date": trade_date,
        "ticker": ticker,
        "name": name,
        "holding_period": holding_period,
        "created_at": datetime.now().isoformat(),
        "trader_attribution": attribution_output
    }
    Path(save_path).mkdir(parents=True, exist_ok=True)
    file_path = Path(save_path) / f"{trade_date}_{ticker}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    return str(file_path)


def load_data(method, period):
    ### 需要进行标准化处理
    predict_data = pd.read_feather(
        os.path.join("records", "normal", str(method), "predict_data.feather"))
    regime_data = pd.read_feather(
        os.path.join("records", "normal", str(method), "regime_data.feather"))
    textuals_data = pd.read_feather(
        os.path.join("records", "normal", str(method),
                     "textuals_data.feather"))
    returns_data = pd.read_feather(
        os.path.join("records", "normal", str(method), "returns_data.feather"))
    returns_data = returns_data[[
        'trade_date', 'code', "nxt1_ret_{0}h".format(period)
    ]]
    predict_data['trade_date'] = pd.to_datetime(predict_data['trade_date'])
    regime_data['trade_date'] = pd.to_datetime(regime_data['trade_date'])
    textuals_data['trade_date'] = pd.to_datetime(textuals_data['trade_date'])
    return predict_data, regime_data, textuals_data, returns_data


def load_foward(method, period):
    dir_path = Path(os.path.join(base_path, "prod", method, str(period)))
    snapshot_dict = {}
    for json_file in dir_path.glob("*.json"):
        date_str = json_file.stem.split("_")[0]
        with open(json_file, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
            snapshot_dict[date_str] = snapshot
    return snapshot_dict


def build_reviewer_params(
        trade_date: str,
        ticker: str,
        name: str,
        holding_period: str,
        regime_str: str,
        predictive_str: str,
        textual_str: str,
        trader_output_dict: dict,
        fwd_ret: float,
        post_t_summary: str = "T+1至T+2持有期内按既定行情结算完成，无额外重大数据修正。") -> dict:
    """
    根据 Trader 预测和真实结算收益，自动计算冻结标签并生成 Reviewer params
    """
    # 确定性计算结算标签 (阈值设定为 ±0.5%)
    theta = 0.005
    if fwd_ret >= theta:
        actual_direction = "UP"
    elif fwd_ret <= -theta:
        actual_direction = "DOWN"
    else:
        actual_direction = "FLAT"

    pred_dir = trader_output_dict.get("predict_direction", "FLAT")
    pred_conf = trader_output_dict.get("confidence", 0.5)

    if pred_dir == "FLAT":
        prediction_outcome = "FLAT_CORRECT" if actual_direction == "FLAT" else "FLAT_MISSED"
    elif actual_direction == "FLAT":
        prediction_outcome = "NEUTRAL"
    else:
        prediction_outcome = "WIN" if pred_dir == actual_direction else "LOSS"

    # 评估置信度是否过度自信
    if prediction_outcome == "LOSS" and pred_conf >= 0.7:
        confidence_assessment = "OVERCONFIDENT"
    elif prediction_outcome == "WIN" and pred_conf <= 0.4:
        confidence_assessment = "UNDERCONFIDENT"
    else:
        confidence_assessment = "JUSTIFIED"

    return {
        "case_id":
        f"CASE-{ticker.split()[0]}-{trade_date}",
        "ticker":
        ticker,
        "name":
        name,
        "prediction_time":
        f"{trade_date} 15:00:00",
        "holding_period":
        holding_period,
        "snapshot_version":
        "v1.0.0",
        "label_thresholds_and_version":
        "Policy: v1.0, Threshold: ±0.5%",
        "regime_features_time_series":
        regime_str,
        "predictive_signals_time_series":
        predictive_str,
        "pre_t_textual_events_data":
        textual_str,
        "pre_prediction_json":
        json.dumps(trader_output_dict, ensure_ascii=False, indent=2),
        "fwd_ret":
        f"{fwd_ret:+.2%}",
        "actual_direction":
        actual_direction,
        "prediction_outcome":
        prediction_outcome,
        "confidence_assessment":
        confidence_assessment,
        "post_t_market_and_event_data":
        post_t_summary
    }


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


async def create_train_agent():
    llm_name = 'deepseek_4001'  ## 指定大模型 包括地址 参数 都存储在对应字典
    vector_name = 'embedding_10002'  ##  指定嵌入模型  包括地址 参数 都存储在对应字典
    persona_name = 'quant_fusion_trader_10001'
    thoughts_name = 'fusion_reviewer_user_100001'
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


async def train(method, period, lookback=3):

    snapshot_dict = load_foward(method=method, period=period)
    predict_data, regime_data, textuals_data, returns_data = await asyncio.to_thread(
        load_data, method=method, period=period)
    train_agent, train_thoughts, train_thoughts_name = await create_train_agent(
    )

    ## 日期交集
    dates = set(predict_data['trade_date']).intersection(
        regime_data['trade_date'], textuals_data['trade_date'])
    dates = [d.strftime('%Y-%m-%d') for d in dates]
    dates.sort()
    #dates = dates[0:lookback + 5]

    ticker = "000852"
    name = '中证1000指数 (000852.SH / IM)'
    holding_period = "T+1开盘 ~ T+{}开盘".format(period + 1)
    semaphore = asyncio.Semaphore(4)
    json_structure = get_fewshot_template2(CaseMemoryReviewResult)
    tasks = []
    save_path = os.path.join(base_path, "attribution", str(method),
                             str(period))
    os.makedirs(save_path, exist_ok=True)
    for index, date in enumerate(dates):
        if index < lookback:
            continue
        end_date = date
        start_date = dates[index - lookback]
        pdata = predict_data[(predict_data['trade_date'] >= start_date)
                             & (predict_data['trade_date'] <= end_date)]
        rdata = regime_data[(regime_data['trade_date'] >= start_date)
                            & (regime_data['trade_date'] <= end_date)]
        tdata = textuals_data[(textuals_data['trade_date'] >= start_date)
                              & (textuals_data['trade_date'] <= end_date)]

        predictive_str = PromptDataBuilder.build_predictive_signals(pdata)
        regime_str = PromptDataBuilder.build_regime_features(rdata)
        textual_str = PromptDataBuilder.build_textual_events(tdata)
        
        fwd_ret_val = returns_data[returns_data['trade_date'] == end_date][
            'nxt1_ret_{0}h'.format(period)].values[0]
        params = build_reviewer_params(
            trade_date=date,
            ticker=ticker,
            name=name,
            holding_period=holding_period,
            regime_str=regime_str,
            predictive_str=predictive_str,
            textual_str=textual_str,
            trader_output_dict=snapshot_dict[date]
            ['trader_prediction'],  # 上一步从 Trader 获得的 output dict
            fwd_ret=fwd_ret_val  # 真实结算收益率
        )
        params["json_structure"] = json_structure

        thought = train_thoughts[train_thoughts_name]
        thougths_cot = thought["cot"]
        prompt = f"""
            {thougths_cot}
            \n\n输出格式必须为:\n
            {{json_structure}}
            """

        tasks.append(
            generate_with_semaphore(trade_time=end_date,
                                    agent_instance=train_agent,
                                    semaphore=semaphore,
                                    human_message=prompt,
                                    params_dict=params,
                                    schema_cls=CaseMemoryReviewResult))

    batch_results = await asyncio.gather(*tasks)
    for result in batch_results:
        save_attribution_snapshot(trade_date=result['output']['trade_time'],
                                  ticker=ticker,
                                  name=name,
                                  holding_period=holding_period,
                                  attribution_output=result['output'],
                                  save_path=save_path)


if __name__ == '__main__':
    method = 'test0'
    asyncio.run(train(method=method, period=3, lookback=3))
