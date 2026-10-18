import pdb
import pandas as pd
import numpy as np
from jdw import DBAPI
from alphacopilot.api.data import DDBAPI, ddb_tools

F2S_MAPPING = {'IF': '000300', 'IH': '000016', 'IC': '000905', 'IM': '000852'}
S2F_MAPPING = {'000300': 'IF', '000016': 'IH', '000905': 'IC', '000852': 'IM'}


class SentimentShock(object):
    """
    情绪脉冲与背离类预测特征 (Sentiment Shock & Divergence Alphas)
    核心逻辑：短期情绪过激、预期差冲击以及期现背离。
    """

    def __init__(self, code, engine=None):
        self.kd_engine = DBAPI.FetchEngine.create_engine('kd') if engine is None else engine
        self.cusomize_api = DDBAPI.cusomize_api()
        self.code = code

    def fetch_spot_daily(self, begin_date, end_date, code):
        """获取指数日线行情"""
        clause_list1 = ddb_tools.to_format('Code', 'in', [code])
        clause_list2 = ddb_tools.to_format(
            'date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format(
            'date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        results = self.cusomize_api.custom(
            table='index_market',
            columns=[
                'date', 'Code', 'openIndex', 'highestIndex', 'lowestIndex',
                'closeIndex', 'turnoverVol', 'turnoverValue'
            ],
            clause_list=[clause_list1, clause_list2, clause_list3],
            format_data=1,
            db_path='tl_daily')
        results.rename(
            columns={
                'date': 'trade_date',
                'Code': 'code',
                'openIndex': 'open',
                'highestIndex': 'high',
                'lowestIndex': 'low',
                'closeIndex': 'close',
                'turnoverVol': 'volume',
                'turnoverValue': 'value'
            },
            inplace=True)
        return results

    def fetch_fut_daily(self, begin_date, end_date, fut_code):
        """获取期货连续合约日线行情"""
        name = 'market_fut'
        names = DBAPI.CustomizeFactory(self.kd_engine).name(name=name)
        clause_list = [
            names.trade_date >= begin_date,
            names.trade_date <= end_date,
            names.contractObject == fut_code
        ]
        daily_market = DBAPI.CustomizeFactory(self.kd_engine).custom(
            name=name, clause_list=clause_list, columns=['trade_date', 'code', 'closePrice'])
        if daily_market.empty:
            return pd.DataFrame()
            
        daily_market.rename(columns={'closePrice': 'fut_close'}, inplace=True)
        # 获取主力合约（通常每日按成交/持仓最大或者最活跃合约，这里按首合约做示例合并）
        main_fut = daily_market.sort_values('trade_date').groupby('trade_date').first().reset_index()
        return main_fut

    def fetch_index_valuation(self, begin_date, end_date):
        """从宏观/估值表获取指数估值水位 (中证500/大盘估值代理)"""
        name = 'fut_macro'
        names = DBAPI.CustomizeFactory(self.kd_engine).name(name=name)
        clause_list = [
            names.periodDate >= begin_date, names.periodDate <= end_date,
            names.indicID.in_(['1170008767'])  # 中证500 TTM PE
        ]
        macro_data = DBAPI.CustomizeFactory(self.kd_engine).custom(
            name=name,
            clause_list=clause_list,
            columns=['indicID', 'publishDate', 'periodDate', 'dataValue'])
        if macro_data.empty:
            return pd.DataFrame()
            
        macro_data['publishDate'] = macro_data['publishDate'].fillna(macro_data['periodDate'])
        macro_data['trade_date'] = pd.to_datetime(macro_data['publishDate']).dt.date
        val_df = macro_data.groupby('trade_date')['dataValue'].last().reset_index()
        val_df.rename(columns={'dataValue': 'pe_ttm'}, inplace=True)
        return val_df

    def _create_valuation_shock(self, val_data, spot_data):
        """
        1. 估值情绪脉冲强度 (Valuation Sentiment Shock Intensity)
        输出连续值特征：
        - price_ret_5d: 5日累计收益率 (连续浮点)
        - price_acc_5d: 收益率二阶导/加速度 (连续浮点)
        - sentiment_shock_intensity: 情绪脉冲强度 (收益率 * 加速度)
        """
        spot_df = spot_data.set_index('trade_date')
        ret_5 = spot_df['close'].pct_change(5)
        ret_acc_5 = ret_5.diff(2)
        
        # 连续情绪冲击强度：当短期涨速与加速度同向时，冲击强度显著放大
        sentiment_shock_intensity = ret_5 * ret_acc_5
        
        return pd.concat([ret_5, ret_acc_5, sentiment_shock_intensity],
                         axis=1,
                         keys=['price_ret_5d', 'price_acc_5d', 'sentiment_shock_intensity'])

    def _create_hf_basis_sentiment(self, spot_data, fut_data):
        """
        2. 期现情绪连续背离度 (Basis Sentiment Divergence Slope)
        输出连续值特征：
        - basis_pct: 基差贴水率 (连续浮点)
        - basis_diff_3d: 3日基差变动斜率 (连续浮点)
        - basis_div_strength: 期现背离连续强度 (-spot_ret * basis_diff)
        """
        if fut_data.empty or spot_data.empty:
            return pd.DataFrame()
            
        merged = pd.merge(spot_data[['trade_date', 'close']], fut_data[['trade_date', 'fut_close']], on='trade_date')
        merged = merged.set_index('trade_date')
        
        basis_pct = (merged['fut_close'] - merged['close']) / merged['close']
        basis_diff_3 = basis_pct.diff(3)
        spot_ret_3 = merged['close'].pct_change(3)
        
        # 连续背离强度：现货下跌但基差走阔(收窄贴水)时为正值，现货上涨但基差走弱时为负值
        basis_div_strength = (-spot_ret_3) * basis_diff_3
        
        return pd.concat([basis_pct, basis_diff_3, basis_div_strength],
                         axis=1,
                         keys=['basis_pct', 'basis_diff_3d', 'basis_div_strength'])

    def start(self, begin_date, end_date):
        spot_df = self.fetch_spot_daily(begin_date, end_date, self.code)
        
        fut_code = S2F_MAPPING.get(self.code, '')
        fut_df = self.fetch_fut_daily(begin_date, end_date, fut_code) if fut_code else pd.DataFrame()
        
        val_df = self.fetch_index_valuation(begin_date, end_date)

        val_shock_data = self._create_valuation_shock(val_df, spot_df)
        basis_sentiment_data = self._create_hf_basis_sentiment(spot_df, fut_df)

        final_features = pd.concat([val_shock_data, basis_sentiment_data], axis=1)
        return final_features
