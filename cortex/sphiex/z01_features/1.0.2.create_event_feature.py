import asyncio, os
import json, pdb
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from kdutils.macro import base_path
from kdutils.ttimes import get_dates



async def create_map_event(method):
    begin_date, end_date = get_dates(method=method)
    from features.text.agent_feature import AgentFeature
    data1 = pd.read_feather(
        os.path.join("records", "basic", method, "total_news.feather"))
    news_data = data1[data1['source'] == 'news']
    news_data = news_data[news_data['date'].between(begin_date,
                                                    end_date,
                                                    inclusive='both')].copy()
    agent_feature = AgentFeature(max_count=2)
    await agent_feature.create_map_event(news_data=news_data)


async def create_reduce_event(method):
    begin_date, end_date = get_dates(method=method)
    from features.text.agent_feature import AgentFeature
    map_dirs = os.path.join(base_path, "data", "event", "map")
    # dates = [
    #     '2026-07-01', '2026-07-02', '2026-07-03', '2026-07-06', '2026-07-07',
    #     '2026-07-08', '2026-07-09', '2026-07-10', '2026-07-13', '2026-07-14',
    #     '2026-07-15', '2026-07-16', '2026-07-17', '2026-07-20', '2026-07-21',
    #     '2026-07-22', '2026-07-23', '2026-07-24', '2026-07-27', '2026-07-28',
    #     '2026-07-29', '2026-07-30', '2026-07-31'
    # ]
    # res = []
    # for d in dates:
    #     ed = pd.read_feather("records/data/event/map/{0}.feather".format(d))
    #     ed['date'] = d
    #     res.append(ed)
    file_path = Path(map_dirs)
    res = []
    for feat_file in file_path.rglob('*.feather'):
        if begin_date <= feat_file.stem <= end_date:
            ed = pd.read_feather(feat_file)
            ed['date'] = feat_file.stem
            res.append(ed)
    event_data = pd.concat(res, axis=0).sort_values(by=['date'])
    agent_feature = AgentFeature(max_count=1)
    await agent_feature.create_reduce_event(event_data=event_data)


async def create_event_feature(method):
    begin_date, end_date = get_dates(method=method)
    ## 加载事件
    from features.text.agent_feature import AgentFeature
    reduce_dirs = os.path.join(base_path, "data", "event", "reduce")
    
    file_path = Path(reduce_dirs)
    res = []
    for feat_file in file_path.rglob('*.feather'):
        if begin_date <= feat_file.stem <= end_date:
            ed = pd.read_feather(feat_file)
            ed['date'] = feat_file.stem
            res.append(ed)
    event_data = pd.concat(res, axis=0).sort_values(by=['date'])
    data1 = pd.read_feather(
        os.path.join(base_path, "basic", method, "total_news.feather"))
    
    gov_data = data1[data1['source'] == 'gov']
    cctv_data = data1[data1['source'] == 'cctv']
    monetary_data = data1[data1['source'] == 'monetary']

    dates = list(event_data['date'].unique())
    agent_feature = AgentFeature(max_count=1)
    await agent_feature.create_event_feature(gov_data=gov_data,
                                             cctv_data=cctv_data,
                                             monetary_data=monetary_data,
                                             event_data=event_data,
                                             dates=dates)


def create_data(method):
    begin_date, end_date = get_dates(method=method)
    from features.text.fetch_data import FetchData
    fetcher = FetchData()
    news_data = fetcher.fetch_news(begin_date, end_date)
    cctv_data = fetcher.fetch_cctv(begin_date, end_date)
    monetary_data = fetcher.fetch_monetary_policy(begin_date, end_date)
    gov_data = fetcher.fetch_gov_policy(begin_date, end_date)
    total_data = pd.concat([news_data, cctv_data, monetary_data, gov_data],
                           axis=0)
    total_data = total_data.sort_values(by=['date']).reset_index(drop=True)
    base_dir = os.path.join("records", "basic", method)
    os.makedirs(base_dir, exist_ok=True)
    pdb.set_trace()
    total_data.to_feather(os.path.join(base_dir, "total_news.feather"))

if __name__ == '__main__':
    method = 'train0'
    #create_data(method=method)
    #asyncio.run(create_map_event(method=method))
    #asyncio.run(create_reduce_event(method=method))
    asyncio.run(create_event_feature(method=method))
    # agg_data()
