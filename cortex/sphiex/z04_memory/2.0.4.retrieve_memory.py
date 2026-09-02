import asyncio, os, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from kdutils.macro import base_path
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


async def run(method, period, lookback):
    ticker = "000852"
    storage_path = os.path.join(base_path, "brain", method, str(period))
    os.makedirs(storage_path, exist_ok=True)
    predict_data, regime_data, textuals_data, _ = await asyncio.to_thread(
        load_data, method=method, period=period)

    dates = set(predict_data['trade_date']).intersection(
        regime_data['trade_date'], textuals_data['trade_date'])
    dates = [d.strftime('%Y-%m-%d') for d in dates]
    dates.sort()
    #dates = dates[0:lookback + 5]

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

        p_martix = pdata[p_cols].values
        r_martix = rdata[r_cols].values

        textual_events = format_textual_events_timeline(tdata)
        
        coordinator.retrieve_experience(
            code=ticker,
            regime_matrix=r_martix,
            predict_matrix=p_martix,
            textual_events=textual_events,
            active_predictive_whitelist={}
        )


if __name__ == '__main__':
    method = 'test0'
    period = 3
    asyncio.run(run(method=method, period=3, lookback=3))
