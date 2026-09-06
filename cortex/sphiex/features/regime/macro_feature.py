import pdb
import pandas as pd
from jdw import DBAPI
from ultron.tradingday import *


class MacroFeature(object):

    def __init__(self, engine=None):
        self.kd_engine = DBAPI.FetchEngine.create_engine(
            'kd') if engine is None else engine
        self._liquidity1_columns = ['1070000007', '1070000009']  # # M1, M2
        self._economic_columns = ['1030000011']  # PMI
        self._credit_columns = ['1070013763']  # 信用周期
        self._industrial_columns = ['1040000702']  # 工业出清
        self._foreign_flow_columns = [
            '1090001399', '3010200075', '1080000004'
        ]  # 外资流向与汇率状态
        self._interest_rate_columns = ['1090001399']  # 绝对利率水位 (资产荒状态)
        self._absolute_valuation_columns = ['1170008767'
                                            ]  # 绝对估值水位 (中证500 TTM PE)
        self._factors_columns = self._liquidity1_columns + self._economic_columns + \
                    self._credit_columns + self._industrial_columns + self._foreign_flow_columns + \
                       self._interest_rate_columns + self._absolute_valuation_columns
        self._factors_columns = list(set(self._factors_columns))

    def fetch_tradingday(self, begin_date, end_date):
        dates = makeSchedule(begin_date, end_date, '1b', 'china.sse',
                             BizDayConventions.Preceding)
        dates = pd.DataFrame({'trade_date': dates})
        return dates

    def load_macroe_data(self, begin_date, end_date, factors_columns):
        name = 'fut_macro'
        names = DBAPI.CustomizeFactory(self.kd_engine).name(name=name)
        clause_list = [
            names.periodDate >= begin_date, names.periodDate <= end_date,
            names.indicID.in_(factors_columns)
        ]
        macro_data = DBAPI.CustomizeFactory(self.kd_engine).custom(
            name=name,
            clause_list=clause_list,
            columns=['indicID', 'publishDate', 'periodDate', 'dataValue'])

        #用 periodDate 填补缺失的 publishDate
        macro_data['publishDate'] = macro_data['publishDate'].fillna(
            macro_data['periodDate'])

        macro_data['publishDate'] = pd.to_datetime(
            macro_data['publishDate']).dt.normalize()
        macro_data = macro_data.drop_duplicates(
            subset=['indicID', 'publishDate', 'periodDate'])
        wide_macro = macro_data.pivot_table(index='publishDate',
                                            columns='indicID',
                                            values='dataValue',
                                            aggfunc='last').reset_index()
        wide_macro = wide_macro.sort_values('publishDate')
        wide_macro = wide_macro[wide_macro.publishDate <= end_date]
        return wide_macro

    def _create_liquidity1(self, base_wide_data):
        liquidity1_data = base_wide_data[
            ['publishDate'] + self._liquidity1_columns].dropna().copy()
        liquidity1_data['m1m2_diff'] = liquidity1_data[
            '1070000007'] - liquidity1_data['1070000009']
        liquidity1_data['m1m2_diff_mom_3m'] = liquidity1_data[
            'm1m2_diff'] - liquidity1_data['m1m2_diff'].shift(3)
        return liquidity1_data[[
            'publishDate', 'm1m2_diff', 'm1m2_diff_mom_3m'
        ]]

    def _create_economic1(self, base_wide_data):
        economic_data = base_wide_data[['publishDate'] +
                                       self._economic_columns].dropna().copy()
        economic_data['pmi_prev'] = economic_data['1030000011'].shift(1)
        return economic_data[['publishDate', 'pmi_prev']]

    def _create_credit1(self, base_wide_data):
        credit_data = base_wide_data[['publishDate'] +
                                     self._credit_columns].dropna().copy()
        credit_data['credit_prev'] = credit_data['1070013763'].shift(1)
        return credit_data[['publishDate', 'credit_prev']]

    def _create_industrial1(self, base_wide_data):
        industrial_data = base_wide_data[
            ['publishDate'] + self._industrial_columns].dropna().copy()
        industrial_data['ppi_prev'] = industrial_data['1040000702'].shift(1)
        industrial_data['ppi_prev_3'] = industrial_data['1040000702'].shift(3)
        return industrial_data[['publishDate', 'ppi_prev', 'ppi_prev_3']]

    def _create_foreign_flow1(self, base_wide_data):
        foreign_flow_data = base_wide_data[['publishDate'] +
                                           self._foreign_flow_columns].copy()
        foreign_flow_data['cn_us_spread'] = foreign_flow_data[
            '1090001399'] - foreign_flow_data['3010200075']
        foreign_flow_data['spread_percentile_252d'] = foreign_flow_data[
            'cn_us_spread'].rolling(window=252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        foreign_flow_data['usd_cny_mom_20d'] = (
            foreign_flow_data['1080000004'] /
            foreign_flow_data['1080000004'].shift(20)) - 1
        return foreign_flow_data[[
            'publishDate', 'cn_us_spread', 'spread_percentile_252d',
            'usd_cny_mom_20d'
        ]]

    def _create_interest_rate1(self, base_wide_data):
        interest_rate_data = base_wide_data[
            ['publishDate'] + self._interest_rate_columns].copy()

        interest_rate_data['cn_10y_percentile_5y'] = interest_rate_data[
            '1090001399'].rolling(window=1260, min_periods=252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        return interest_rate_data[['publishDate', 'cn_10y_percentile_5y']]

    def _create_absolute_valuation1(self, base_wide_data):
        absolute_valuation_data = base_wide_data[
            ['publishDate'] + self._absolute_valuation_columns].copy()

        absolute_valuation_data['pe_500_percentile'] = absolute_valuation_data[
            '1170008767'].rolling(window=1260, min_periods=252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        absolute_valuation_data[
            'pe_500_sample_size'] = absolute_valuation_data[
                '1170008767'].rolling(window=1260, min_periods=252).count()
        return absolute_valuation_data[[
            'publishDate', 'pe_500_percentile', 'pe_500_sample_size'
        ]]

    def start(self, begin_date, end_date):
        start_date = advanceDateByCalendar('china.sse', begin_date, '-256b')
        wide_macro = self.load_macroe_data(
            begin_date=start_date,
            end_date=end_date,
            factors_columns=self._factors_columns)
        nofill_wide_macro = wide_macro.copy()
        fill_wid_macro = wide_macro.ffill()
        ## 低频特征
        liquidity1_data = self._create_liquidity1(
            base_wide_data=nofill_wide_macro)
        economic1_data = self._create_economic1(
            base_wide_data=nofill_wide_macro)
        credit1_data = self._create_credit1(base_wide_data=nofill_wide_macro)
        industrial1_data = self._create_industrial1(
            base_wide_data=nofill_wide_macro)
        
        ## 高频特征
        foreignflow1_data = self._create_foreign_flow1(
            base_wide_data=fill_wid_macro)
        interest_rate1_data = self._create_interest_rate1(
            base_wide_data=fill_wid_macro)
        absolute_valuation1_data = self._create_absolute_valuation1(
            base_wide_data=fill_wid_macro)

        dates_data = self.fetch_tradingday(begin_date=begin_date,
                                           end_date=end_date)

        # 3. 将所有特征模块横向拼接回主干宽表
        feature_dfs = [
            liquidity1_data, economic1_data, credit1_data, industrial1_data,
            foreignflow1_data, interest_rate1_data, absolute_valuation1_data
        ]
        
        all_features = pd.concat([
            df.set_index('publishDate') for df in feature_dfs if df is not None
        ],
                                 axis=1)
        all_features = all_features.ffill()
        aligned_df = pd.merge_asof(
            dates_data.set_index('trade_date'),
            all_features,
            left_index=True,
            right_index=True,
            direction='backward',
            allow_exact_matches=False).reset_index()
        aligned_df = aligned_df.ffill()
        return aligned_df
