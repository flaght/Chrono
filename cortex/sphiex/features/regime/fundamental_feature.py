import pdb
import pandas as pd
from jdw import DBAPI
from alphacopilot.api.data import DDBAPI, ddb_tools

F2S_MAPPING = {'IF': '000300', 'IH': '000016', 'IC': '000905', 'IM': '000852'}
S2F_MAPPING = {'000300': 'IF', '000016': 'IH', '000905': 'IC', '000852': 'IM'}

class FundamentalFeature(object):
    def __init__(self, code, engine=None):
        self.kd_engine = DBAPI.FetchEngine.create_engine('kd') if engine is None else engine
        self.cusomize_api = DDBAPI.cusomize_api()
        self.code = code

    def fetch_index_weights(self, begin_date, end_date):
        """获取指数成分股及其权重"""
        clause_list1 = ddb_tools.to_format('indexCode', 'in', [self.code])
        clause_list2 = ddb_tools.to_format('date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format('date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        
        results = self.cusomize_api.custom(
            table='index_components',
            columns=['date', 'Code', 'weight'],
            clause_list=[clause_list1, clause_list2, clause_list3],
            format_data=1,
            db_path='tl_daily'
        )
        results.rename(columns={'date': 'trade_date', 'Code': 'code'}, inplace=True)
        return results

    def fetch_fin_maindata(self, begin_date, end_date, codes):
        """获取个股净利润同比增速 (Point in time)"""
        clause_list1 = ddb_tools.to_format('date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list2 = ddb_tools.to_format('date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format('Code', 'in', codes)
        
        results = self.cusomize_api.custom(
            table='fin_maindata',
            columns=['date', 'Code', 'niAttrPYoy'],
            clause_list=[clause_list1, clause_list2, clause_list3],
            format_data=1,
            db_path='tl_daily'
        )
        results.rename(columns={'date': 'trade_date', 'Code': 'code'}, inplace=True)
        return results

    def _create_fundamental_growth(self, base_data):
        """3.8 盈利拐点特征 (Fundamental Growth Regime)
        基于指数成分股加权净利润同比增速的二阶导数
        """
        if base_data.empty or 'niAttrPYoy' not in base_data.columns:
            return pd.DataFrame()
            
        # 1. 计算每日指数加权的净利润同比增速
        base_data = base_data.dropna(subset=['niAttrPYoy', 'weight'])
        
        # 权重归一化 (防止部分成分股数据缺失导致总权重不为 100)
        base_data['weight_norm'] = base_data.groupby('trade_date')['weight'].transform(lambda x: x / x.sum())
        base_data['weighted_yoy'] = base_data['niAttrPYoy'] * base_data['weight_norm']
        
        index_yoy = base_data.groupby('trade_date')['weighted_yoy'].sum().reset_index()
        index_yoy = index_yoy.set_index('trade_date')
        
        # 2. 计算二阶导数 (当期 YoY - 上期 YoY)
        # 财报大多季报更新，取过去 60 个交易日（一季度）的差值衡量拐点动量
        index_yoy['yoy_momentum_60'] = index_yoy['weighted_yoy'].diff(60)
        
        return index_yoy[['yoy_momentum_60']]

    def start(self, begin_date, end_date):
        weights_df = self.fetch_index_weights(begin_date, end_date)
        if weights_df.empty:
            return pd.DataFrame()
            
        codes = weights_df['code'].unique().tolist()
        fin_df = self.fetch_fin_maindata(begin_date, end_date, codes)
        if fin_df.empty:
            return pd.DataFrame()
        # 1. 财报同日发布去重 (保留最新公布值)
        fin_df = fin_df.dropna(subset=['niAttrPYoy']).drop_duplicates(
            subset=['trade_date', 'code'], keep='last'
        )
        # 2. 将低频财报数据展开为全交易日 Panel 宽表并 ffill (Point-in-Time)
        trading_days = weights_df['trade_date'].drop_duplicates().sort_values()
        
        # 构建 (trade_date, code) 完整的时序网格
        fin_pivot = fin_df.pivot(index='trade_date', columns='code', values='niAttrPYoy')
        # 对齐到所有交易日并前向填充 (财报公布后持续有效)
        fin_aligned = fin_pivot.reindex(trading_days).ffill()
        fin_long = fin_aligned.stack().reset_index()
        fin_long.columns = ['trade_date', 'code', 'niAttrPYoy']
        # 3. 与每日成分股权重进行 merge
        market_data = pd.merge(weights_df, fin_long, on=['trade_date', 'code'], how='inner')
        fundamental_growth_data = self._create_fundamental_growth(market_data)
        return fundamental_growth_data