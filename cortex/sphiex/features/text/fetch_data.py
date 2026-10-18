import pdb, re
import pandas as pd
from alphacopilot.api.data import DDBAPI, ddb_tools
from ultron.tradingday import *


def should_keep(row):
    channel = row["channels"]
    score = row["relevance_score"]

    if channel in {"央行", "宏观", "两会"}:
        return score >= 3

    if channel in {"数据", "国际"}:
        return score >= 4

    if channel in {"焦点", "市场", "A股", "行业", "原创", "其他", ""}:
        return score >= 5

    if channel in {"公司", "观点"}:
        return score >= 7

    return False


CHANNEL_BASE_SCORE = {
    "央行": 6,
    "宏观": 5,
    "两会": 5,
    "数据": 4,
    "国际": 3,
    "焦点": 2,
    "市场": 2,
    "A股": 2,
    "行业": 1,
    "原创": 1,
    "其他": 0,
    "": 0,
    "观点": -2,
    "公司": -3,
}

HIGH_VALUE_PATTERNS = [
    # 中国货币政策
    r"央行",
    r"人民银行",
    r"降准",
    r"降息",
    r"LPR",
    r"MLF",
    r"逆回购",
    r"公开市场操作",
    r"流动性",

    # 国内高层政策
    r"国务院",
    r"中央政治局",
    r"中央财经委员会",
    r"发改委",
    r"财政部",
    r"证监会",
    r"金融监管总局",
    r"专项债",
    r"房地产政策",

    # 国内外宏观数据
    r"\bCPI\b",
    r"\bPPI\b",
    r"\bPMI\b",
    r"\bGDP\b",
    r"非农",
    r"失业率",
    r"通胀",
    r"社融",
    r"新增贷款",
    r"进出口",

    # 全球货币政策
    r"美联储",
    r"FOMC",
    r"鲍威尔",
    r"欧洲央行",
    r"日本央行",
    r"利率决议",

    # 系统性事件
    r"关税",
    r"制裁",
    r"冲突",
    r"停火",
    r"战争",
    r"债务违约",
    r"银行危机",
    r"金融风险",
]

MEDIUM_VALUE_PATTERNS = [
    r"人民币",
    r"美元指数",
    r"美债收益率",
    r"国债收益率",
    r"布伦特原油",
    r"WTI原油",
    r"黄金",
    r"铜价",
    r"铁矿石",
    r"大宗商品",
    r"中概股",
    r"纳斯达克中国金龙指数",
]

NOISE_PATTERNS = [
    r"目标价从.*(?:上调|下调)至",
    r"评级从.*(?:上调|下调)",
    r"增减持汇总",
    r"龙虎榜",
    r"融资买入",
    r"融券卖出",
    r"大宗交易",
    r"财报提醒",
    r"发布业绩报告",
    r"盘中涨超",
    r"盘中跌超",
    r"涨停",
    r"跌停",
    r"回购.*股份",
    r"IPO.*估值",
]


def count_matches(text, patterns):
    return sum(
        bool(re.search(pattern, text, flags=re.I)) for pattern in patterns)


def calculate_relevance_score(row):
    text = row["text"]
    channel = row["channels"]

    score = CHANNEL_BASE_SCORE.get(channel, 0)

    score += count_matches(text, HIGH_VALUE_PATTERNS) * 3
    score += count_matches(text, MEDIUM_VALUE_PATTERNS) * 1
    score -= count_matches(text, NOISE_PATTERNS) * 5

    return score


class FetchData(object):

    def __init__(self):
        self.cusomize_api = DDBAPI.cusomize_api()

    def fetch_news(self, begin_date, end_date):
        pdb.set_trace()
        EXCLUDE_PATTERNS = [
            r"目标价从.*(?:上调|下调)至",
            r"评级从.*(?:上调|下调)",
            r"增减持汇总",
            r"龙虎榜",
            r"融资买入",
            r"融券卖出",
            r"大宗交易",
            r"财报提醒",
            r"发布业绩报告",
            r"盘中涨超",
            r"盘中跌超",
            r"涨停",
            r"跌停",
        ]
        clause_list2 = ddb_tools.to_format(
            'date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format(
            'date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        results = self.cusomize_api.custom(
            table='news',
            columns=['date', 'datetime', 'title', 'content', 'channels'],
            clause_list=[clause_list2, clause_list3],
            format_data=1,
            db_path='ts_daily')
        results.rename({'datetime': 'publish_time'}, inplace=True)
        results['content'] = results['title'] + ' ' + results['content']
        results['date'] = pd.to_datetime(
            results['date']).dt.strftime('%Y-%m-%d')
        results[(results['date'] >= begin_date)
                & (results['date'] <= end_date)]
        results['source'] = 'news'
        results["channels"] = results["channels"].fillna("").str.strip()
        results["text"] = (
            results["title"].fillna("").str.strip() + " " +
            results["content"].fillna("").str.strip()).str.strip()
        results = results[results["text"].str.len() >= 15]

        results["relevance_score"] = results.apply(
            calculate_relevance_score,
            axis=1,
        )
        exclude_regex = "|".join(EXCLUDE_PATTERNS)
        results = results[~results["content"].str.
                          contains(exclude_regex, regex=True, na=False)]
        results = results[results.apply(should_keep, axis=1)].copy()
        results = results[['date', 'datetime',
                           'text']].rename(columns={
                               'datetime': 'publish_time',
                               'text': 'content'
                           })
        results['source'] = 'news'
        return results

    def fetch_cctv(self, begin_date, end_date):
        clause_list2 = ddb_tools.to_format(
            'date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format(
            'date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        results = self.cusomize_api.custom(
            table='cctv_news',
            columns=['date', 'title', 'content'],
            clause_list=[clause_list2, clause_list3],
            format_data=1,
            db_path='ts_daily')
        results['publish_time'] = results['date']
        results['date'] = pd.to_datetime(
            results['date']).dt.strftime('%Y-%m-%d')
        results[(results['date'] >= begin_date)
                & (results['date'] <= end_date)]

        results['source'] = 'cctv'
        results = results.drop(['title'], axis=1)
        return results

    def fetch_gov_policy(self, begin_date, end_date):
        clause_list2 = ddb_tools.to_format(
            'date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format(
            'date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        results = self.cusomize_api.custom(
            table='gov_policy',
            columns=['date', 'pubtime', 'title', 'content'],
            clause_list=[clause_list2, clause_list3],
            format_data=1,
            db_path='ts_daily')
        results['date'] = pd.to_datetime(
            results['date']).dt.strftime('%Y-%m-%d')
        results[(results['date'] >= begin_date)
                & (results['date'] <= end_date)]
        results.rename(columns={'pubtime':'publish_time'},inplace=True)
        results['content'] = results['title'] + ' ' + results['content']
        results['source'] = 'gov'
        results = results.drop(['title'], axis=1)
        return results

    def fetch_monetary_policy(self, begin_date, end_date):
        clause_list2 = ddb_tools.to_format(
            'date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format(
            'date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        results = self.cusomize_api.custom(
            table='monetary_policy',
            columns=['date', 'pubtime', 'title', 'content'],
            clause_list=[clause_list2, clause_list3],
            format_data=1,
            db_path='ts_daily')
        results['date'] = pd.to_datetime(
            results['date']).dt.strftime('%Y-%m-%d')
        results[(results['date'] >= begin_date)
                & (results['date'] <= end_date)]
        results['content'] = results['title'] + ' ' + results['content']
        results['source'] = 'monetary'
        results = results.drop(['title'], axis=1)
        return results
