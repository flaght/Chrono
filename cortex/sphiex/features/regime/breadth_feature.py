import pdb
import pandas as pd
from jdw import DBAPI
from alphacopilot.api.data import DDBAPI, ddb_tools

#MAPPING_CODE = {'IF': '000300', 'IH': '000016', 'IC': '000905', 'IM': '000852'}

F2S_MAPPING = {'IF': '000300', 'IH': '000016', 'IC': '000905', 'IM': '000852'}
S2F_MAPPING = {'000300': 'IF', '000016': 'IH', '000905': 'IC', '000852': 'IM'}


class BreadthFeature(object):
    def __init__(self, code, engine=None):
        self.kd_engine = DBAPI.FetchEngine.create_engine('kd') if engine is None else engine
        self.cusomize_api = DDBAPI.cusomize_api()
        self.code = code

    def fetch_index_components(self, begin_date, end_date):
        """获取指数成分股"""
        clause_list1 = ddb_tools.to_format('indexCode', 'in', [self.code])
        clause_list2 = ddb_tools.to_format('date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format('date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        
        results = self.cusomize_api.custom(
            table='index_components',
            columns=['date', 'Code'],
            clause_list=[clause_list1, clause_list2, clause_list3],
            format_data=1,
            db_path='tl_daily'
        )
        results.rename(columns={'date': 'trade_date', 'Code': 'code'}, inplace=True)
        return results

    def fetch_stock_market(self, begin_date, end_date, codes):
        """获取成分股的日线行情"""
        # 注意: 预留更多历史数据以计算个股涨跌幅，假设调用 MktEqudGet (表名 market_equd)
        clause_list1 = ddb_tools.to_format('date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list2 = ddb_tools.to_format('date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format('Code', 'in', codes)
        results = self.cusomize_api.custom(
            table='market_stock',
            columns=['date', 'Code', 'closePrice'],
            clause_list=[clause_list1, clause_list2, clause_list3],
            format_data=1,
            db_path='tl_daily'
        )
        results.rename(columns={'date': 'trade_date', 'Code': 'code', 'closePrice': 'close'}, inplace=True)
        return results

    def _create_market_breadth(self, base_data):
        """12. 市场宽度特征 (Market Breadth Regime)
        计算成分股上涨比例，及其在历史的滚动分位数
        """
        if base_data.empty or 'close' not in base_data.columns:
            return pd.DataFrame()
            
        # 1. 计算个股的每日涨跌幅
        base_data = base_data.sort_values(['code', 'trade_date'])
        base_data['ret'] = base_data.groupby('code')['close'].pct_change()
        
        # 2. 标记是否上涨
        base_data['is_up'] = (base_data['ret'] > 0).astype(int)
        
        # 3. 按日聚合上涨比例
        valid_data = base_data.dropna(subset=['ret'])
        breadth_ratio = valid_data.groupby('trade_date')['is_up'].mean().reset_index()
        breadth_ratio = breadth_ratio.set_index('trade_date')
        
        # 4. 计算滚动分位数
        breadth_percentile_252 = breadth_ratio['is_up'].rolling(
            window=252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
                
        breadth_df = pd.concat([breadth_ratio['is_up'], breadth_percentile_252], 
                               axis=1, 
                               keys=['breadth_ratio', 'breadth_percentile_252'])
        return breadth_df

    def start(self, begin_date, end_date):
        comp_df = self.fetch_index_components(begin_date, end_date)
        if comp_df.empty:
            return pd.DataFrame()
            
        codes = comp_df['code'].unique().tolist()
        market_df = self.fetch_stock_market(begin_date, end_date, codes)
        if market_df.empty:
            return pd.DataFrame()
            
        valid_market = pd.merge(comp_df, market_df, on=['trade_date', 'code'], how='inner')
        breadth_data = self._create_market_breadth(valid_market)
        
        return breadth_data
