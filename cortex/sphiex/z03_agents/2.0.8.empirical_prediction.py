import asyncio, os, pdb, json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from features.text.mode import TraderPredictionResult
from lib.inkits.utils import build_dynamic_schema, get_fewshot_template2
from kdutils.macro import base_path
from lib.agt001 import create_agent
from lib.pdb001 import PromptDataBuilder
from lib.amr001 import MARLMemoryCoordinator


# 中文类型映射字典，增强可读性
FEATURE_TYPE_MAP = {
    "domestic_macro": "国内宏观",
    "domestic_policy": "国内政策",
    "global_liquidity": "全球流动性",
    "external_shock": "外部冲击",
    "industry_trend": "行业趋势",
    "market_sentiment": "市场情绪"
}


def save_prediction_snapshot(trade_date, ticker, name, holding_period,
                             input_data_dict, prediction_output,
                             forward_return, save_path):
    snapshot = {
        "sample_id": f"SMP-{ticker}-{trade_date}",
        "trade_date": trade_date,
        "ticker": ticker,
        "name": name,
        "holding_period": holding_period,
        "created_at": datetime.now().isoformat(),
        # 原始输入数据快照 (Point-in-Time 冻结)
        "input_context": {
            "regime_features_table":
            input_data_dict.get("regime_features_time_series"),
            "predictive_signals_table":
            input_data_dict.get("predictive_signals_time_series"),
            "textual_events_timeline":
            input_data_dict.get("textual_events_data"),
        },
        # Trader 预测结果
        "trader_prediction": prediction_output,
        # 状态标记：等待真实收益结算
        "status": "SETTLED",
        "forward_return": forward_return,
        "reviewer_result": ""
    }
    Path(save_path).mkdir(parents=True, exist_ok=True)
    file_path = Path(save_path) / f"{trade_date}_{ticker}.json"
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
            return {
                "output": output,
                "status": 0,
                "input": params_dict
            }
        except Exception as e:
            print(f"❌ [{agent_instance.name}] 生成期间发生错误: {e}")
            return {"output": "", "status": -1}


def load_data(method, period):
    ### 需要进行标准化处理
    predict_data = pd.read_feather(
        os.path.join("records", "normal", str(method), "predict_data.feather"))
    regime_data = pd.read_feather(
        os.path.join("records", "normal", str(method), "regime_data.feather"))
    textuals_data = pd.read_feather(
        os.path.join("records", "normal", str(method),
                     "textuals_data.feather"))

    predict_data['trade_date'] = pd.to_datetime(predict_data['trade_date'])
    regime_data['trade_date'] = pd.to_datetime(regime_data['trade_date'])
    textuals_data['trade_date'] = pd.to_datetime(textuals_data['trade_date'])
    return predict_data, regime_data, textuals_data


def format_textual_events_timeline(events_data,
                                   current_trade_date=None) -> str:
    base_line = "【重大事件】"
    if events_data is None or (isinstance(
            events_data, pd.DataFrame) and events_data.empty) or (isinstance(
                events_data, list) and len(events_data) == 0):
        return "{0}\n".format(base_line)
    df = events_data.copy()

    df['trade_date'] = df['trade_date'].astype(str)
    df = df.sort_values(by=['trade_date', 'importance'],
                        ascending=[True, False])

    unique_dates = sorted(df['trade_date'].unique().tolist())
    if current_trade_date is None:
        target_date = unique_dates[-1]
    else:
        target_date = str(current_trade_date)
    lines = [base_line]
    # 4. 按日期分层聚合
    for d in unique_dates:
        # 计算相对 T 日的标签
        if d == target_date:
            date_label = f"[{d} (T日)]:"
        else:
            # 计算距离当前日期的倒推相对天数序号（或简单标记历史）
            idx_from_end = len(unique_dates) - 1 - unique_dates.index(d)
            t_tag = f"T-{idx_from_end}日" if idx_from_end > 0 else "历史"
            date_label = f"[({t_tag})]:"
        lines.append(date_label)

        # 提取当天的事件
        day_events = df[df['trade_date'] == d]
        for _, row in day_events.iterrows():
            raw_type = str(row['feature_type'])
            type_cn = FEATURE_TYPE_MAP.get(raw_type, raw_type)
            summary = str(row['summary'])[:400].strip()

            lines.append(f"- [{type_cn}] {summary}")

        lines.append("")  # 空行分隔日期段
    return "\n".join(lines).strip()


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


async def run(method, period, lookback):
    ticker = "000852"
    name = '中证1000指数 (000852.SH / IM)'
    holding_period = "T+1开盘 ~ T+{}开盘".format(period + 1)
    semaphore = asyncio.Semaphore(4)
    tasks = []
    storage_path = os.path.join(base_path, "brain", method, str(period))
    save_path = os.path.join(base_path, "enhanced", str(method), str(period))
    os.makedirs(storage_path, exist_ok=True)
    predict_data, regime_data, textuals_data = await asyncio.to_thread(
        load_data, method=method, period=period)

    predict_agent, predict_thoughts, predict_thoughts_name = await create_predict_agent(
    )

    p_cols = [
        f for f in predict_data.columns if not f in ['trade_date', 'code']
    ]
    r_cols = [
        f for f in regime_data.columns if not f in ['trade_date', 'code']
    ]
    p_dim = len(p_cols) * (lookback + 1)
    r_dim = len(r_cols) * (lookback + 1)

    coordinator = MARLMemoryCoordinator(name="ashare_{0}".format(ticker),
                                        storage_path=storage_path,
                                        vector_provider='fassis',
                                        embedding_model='text-embedding-v4',
                                        embedding_provider='openai',
                                        p_dim=p_dim,
                                        r_dim=r_dim)

    dates = set(predict_data['trade_date']).intersection(
        regime_data['trade_date'], textuals_data['trade_date'])
    dates = [d.strftime('%Y-%m-%d') for d in dates]
    dates.sort()
    dates = dates
    for index, date in enumerate(dates):
        if index < lookback:
            continue
        pdb.set_trace()
        end_date = date
        start_date = dates[index - lookback]
        pdata = predict_data[(predict_data['trade_date'] >= start_date)
                             & (predict_data['trade_date'] <= end_date)]
        rdata = regime_data[(regime_data['trade_date'] >= start_date)
                            & (regime_data['trade_date'] <= end_date)]
        tdata = textuals_data[(textuals_data['trade_date'] >= start_date)
                              & (textuals_data['trade_date'] <= end_date)]

        p_martix = pdata[p_cols].values
        r_martix = rdata[r_cols].values
        textual_events = format_textual_events_timeline(tdata)
        

        ## 经验检索
        memories_str = coordinator.retrieve_experience(
            code=ticker,
            regime_matrix=r_martix,
            predict_matrix=p_martix,
            textual_events=textual_events,
            active_predictive_whitelist={})
        pdb.set_trace()
        predictive_str = PromptDataBuilder.build_predictive_signals(pdata)
        regime_str = PromptDataBuilder.build_regime_features(rdata)
        textual_str = PromptDataBuilder.build_textual_events(tdata)

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
    batch_results = await asyncio.gather(*tasks)
    for result in batch_results:
        save_prediction_snapshot(trade_date=result['output']['trade_time'],
                                 ticker=ticker,
                                 name=name,
                                 holding_period=holding_period,
                                 input_data_dict=result['input'],
                                 prediction_output=result['output'],
                                 forward_return=result['forward_return'],
                                 save_path=save_path)


if __name__ == '__main__':
    method = 'test0'
    period = 3
    asyncio.run(run(method=method, period=3, lookback=3))