import pdb, os, asyncio, datetime
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro import base_path


def create_base_features(begin_date, end_date, code='000852'):
    from create_event_features import create_features as create_event_features
    from create_regime_features import create_features as create_regime_features
    from create_predict_features import create_features as create_predict_features
    pdb.set_trace()
    predict_data = create_predict_features(begin_date=begin_date,
                                           end_date=end_date,
                                           code=code)

    regime_data = create_regime_features(begin_date=begin_date,
                                         end_date=end_date,
                                         code=code)
    evnet_data = create_event_features(begin_date=begin_date,
                                       end_date=end_date)

    base_dir = os.path.join("records", "live", "temp", end_date, "basic")
    os.makedirs(base_dir, exist_ok=True)

    predict_data.to_feather(os.path.join(base_dir, "predict_data.feather"))
    regime_data.to_feather(os.path.join(base_dir, "regime_data.feather"))
    evnet_data.to_feather(os.path.join(base_dir, "event_data.feather"))


async def create_map_event(begin_date, end_date):
    base_dir = os.path.join("records", "live", "temp", end_date, 'basic')
    from features.text.agent_feature import AgentFeature
    data1 = pd.read_feather(os.path.join(base_dir, "event_data.feather"))
    news_data = data1[data1['source'] == 'news']
    news_data = news_data[news_data['date'].between(begin_date,
                                                    end_date,
                                                    inclusive='both')].copy()
    agent_feature = AgentFeature(max_count=1)
    await agent_feature.create_map_event(news_data=news_data)


async def create_reduce_event(begin_date, end_date):
    from features.text.agent_feature import AgentFeature
    map_dirs = os.path.join(base_path, "data", "event", "map")
    file_path = Path(map_dirs)
    res = []
    for feat_file in file_path.rglob('*.feather'):
        print(feat_file.stem)
        if begin_date <= feat_file.stem <= end_date:
            ed = pd.read_feather(feat_file)
            ed['date'] = feat_file.stem
            res.append(ed)
    event_data = pd.concat(res, axis=0).sort_values(by=['date'])
    agent_feature = AgentFeature(max_count=1)
    await agent_feature.create_reduce_event(event_data=event_data)


async def create_event_feature(begin_date, end_date):
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
        os.path.join("records", "live", "temp", end_date, 'basic',
                     "event_data.feather"))

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


def create_aggregation_feature(begin_date, end_date):
    asyncio.run(create_map_event(begin_date=begin_date, end_date=end_date))
    asyncio.run(create_reduce_event(begin_date=begin_date, end_date=end_date))
    asyncio.run(create_event_feature(begin_date=begin_date, end_date=end_date))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",
                        choices=["basic", "aggregation"],
                        default="aggregation")
    
    args = parser.parse_args()
    #for date in ['2026-09-21','2026-09-22','2026-09-23','2026-09-24','2026-09-25','2026-09-28','2026-09-29']:
    #date = '2026-10-09'
    #begin_date = end_date = advanceDateByCalendar('china.sse',
    #                                              datetime.datetime.now(),
    #                                              '-1b').strftime('%Y-%m-%d')

    
    if args.method == 'aggregation':
        from lib.calendars.api import *
        begin_date = advanceDateByCalendar('china.sse',
                                       datetime.datetime.now(),
                                       '-1b').strftime('%Y-%m-%d')
        end_date = begin_date
        dates = makeSchedule(begin_date, end_date, '1b', 'china.sse',
                             BizDayConventions.Preceding)
    else:
        from ultron.tradingday import *
        begin_date = advanceDateByCalendar('china.sse',
                                       datetime.datetime.now(),
                                       '-1b').strftime('%Y-%m-%d')
        end_date = begin_date
        dates = makeSchedule(begin_date, end_date, '1b', 'china.sse',
                             BizDayConventions.Preceding)
    for date in dates:
        begin_date = end_date = date.strftime('%Y-%m-%d')
        print(begin_date, end_date)
        if args.method == 'aggregation':
            create_aggregation_feature(begin_date=begin_date,
                                       end_date=end_date)
        else:
            create_base_features(begin_date=begin_date, end_date=end_date)
