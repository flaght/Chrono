import pdb
import pandas as pd
import numpy as np
from jdw import DBAPI
from alphacopilot.api.data import DDBAPI, ddb_tools


class OverviewFeature(object):

    def __init__(self, code='000852', engine=None):
        self.kd_engine = DBAPI.FetchEngine.create_engine(
            'kd') if engine is None else engine
        self.cusomize_api = DDBAPI.cusomize_api()
        self.code = code  # 目标指数代码，此处仅用于兼容框架

    def fetch_index_market(self, begin_date, end_date):
        """获取全市场重要指数日线行情"""
        # 我们需要上证(000001)、深证(399001)、沪深300(000300)、中证1000(000852)
        codes = ['000001', '399001', '000300', '000852']

        clause_list1 = ddb_tools.to_format('Code', 'in', codes)
        clause_list2 = ddb_tools.to_format(
            'date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format(
            'date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))

        results = self.cusomize_api.custom(
            table='index_market',
            columns=['date', 'Code', 'closeIndex', 'turnoverValue'],
            clause_list=[clause_list1, clause_list2, clause_list3],
            format_data=1,
            db_path='tl_daily')
        results.rename(columns={
            'date': 'trade_date',
            'Code': 'code',
            'closeIndex': 'close',
            'turnoverValue': 'value'
        },
                       inplace=True)
        return results

    def fetch_limit_stats(self, begin_date, end_date):
        """获取全市场涨跌停数据"""
        clause_list1 = ddb_tools.to_format(
            'date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list2 = ddb_tools.to_format(
            'date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))

        try:
            results = self.cusomize_api.custom(
                table='mkt_limit_stats',
                columns=['date', 'statsCode', 'upLimitNum', 'downLimitNum'],
                clause_list=[clause_list1, clause_list2],
                format_data=1,
                db_path='tl_daily')
            results.rename(columns={'date': 'trade_date'}, inplace=True)
            return results
        except Exception:
            # 如果表不存在，返回空
            return pd.DataFrame()

    def _create_total_turnover(self, index_data):
        """1. 两市合计成交量能"""
        # 过滤上证和深证
        market_idx = index_data[index_data['code'].isin(['000001', '399001'])]

        # 按日汇总成交金额
        total_turnover = market_idx.groupby(
            'trade_date')['value'].sum().reset_index()
        total_turnover.set_index('trade_date', inplace=True)

        # 计算 252 日滚动分位数
        # 为了避免前期数据不足，min_periods 设置为 60
        turnover_pct_252 = total_turnover['value'].rolling(
            window=252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        return pd.concat(
            [total_turnover['value'], turnover_pct_252],
            axis=1,
            keys=['total_market_turnover', 'turnover_percentile_252'])

    def _create_style_spread(self, index_data):
        """2. 跨指数风格剪刀差 (中证1000 vs 沪深300)"""
        # 过滤 000852 和 000300
        style_idx = index_data[index_data['code'].isin(['000300', self.code])]
        if style_idx.empty:
            return pd.DataFrame()

        # 转换为宽表
        wide_style = style_idx.pivot(index='trade_date',
                                     columns='code',
                                     values='close')

        if self.code not in wide_style.columns or '000300' not in wide_style.columns:
            return pd.DataFrame()

        # 计算日收益率
        ret_1000 = wide_style[self.code].pct_change()
        ret_300 = wide_style['000300'].pct_change()

        # 计算剪刀差 (中小盘跑赢大盘幅度)
        size_spread = ret_1000 - ret_300

        return pd.concat([size_spread], axis=1, keys=['size_style_spread'])

    def _create_limit_ratio(self, limit_data):
        """3. 全市场多空比 (涨停跌停比)"""
        if limit_data.empty:
            return pd.DataFrame()

        # 这里为了稳健起见，尝试按日期求最大值作为全市场总和
        # 实际情况可根据 Uqer 真实数据微调
        daily_limit = limit_data.groupby('trade_date')[[
            'upLimitNum', 'downLimitNum'
        ]].max()

        # 多空比 = 涨停数 / (跌停数 + 1)
        up_down_ratio = daily_limit['upLimitNum'] / (
            daily_limit['downLimitNum'] + 1)

        # 计算极端多头的历史分位
        ratio_pct_252 = up_down_ratio.rolling(
            window=252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        return pd.concat([
            daily_limit['upLimitNum'], daily_limit['downLimitNum'],
            up_down_ratio, ratio_pct_252
        ],
                         axis=1,
                         keys=[
                             'up_limit_num', 'down_limit_num', 'up_down_ratio',
                             'up_down_ratio_pct_252'
                         ])

    def start(self, begin_date, end_date):
        # 1. 获取数据
        index_data = self.fetch_index_market(begin_date, end_date)
        limit_data = self.fetch_limit_stats(begin_date, end_date)

        if index_data.empty:
            return pd.DataFrame()

        # 2. 计算各维度特征
        turnover_df = self._create_total_turnover(index_data)
        spread_df = self._create_style_spread(index_data)
        limit_df = self._create_limit_ratio(limit_data)

        # 3. 合并特征
        feature_dfs = [
            df for df in [turnover_df, spread_df, limit_df] if not df.empty
        ]
        if not feature_dfs:
            return pd.DataFrame()

        overview_data = pd.concat(feature_dfs, axis=1)

        return overview_data
