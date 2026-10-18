import pdb
import pandas as pd
import numpy as np
from jdw import DBAPI
from alphacopilot.api.data import DDBAPI, ddb_tools

F2S_MAPPING = {'IF': '000300', 'IH': '000016', 'IC': '000905', 'IM': '000852'}
S2F_MAPPING = {'000300': 'IF', '000016': 'IH', '000905': 'IC', '000852': 'IM'}


class SmartMoney(object):
    """
    主力资金与筹码博弈类预测特征 (Smart Money & Position Flow Alphas)
    核心逻辑：捕捉机构、“聪明钱”和套保席位的提前布局与异常异动。
    """

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

    def fetch_moneyflow(self, begin_date, end_date, codes):
        """获取成分股日度大单与超大单资金流向 (market_moneyflow)"""
        clause_list1 = ddb_tools.to_format('date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list2 = ddb_tools.to_format('date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format('Code', 'in', codes)
        
        results = self.cusomize_api.custom(
            table='market_moneyflow',
            columns=['date', 'Code', 'mainFlow', 'turnoverValue'],
            clause_list=[clause_list1, clause_list2, clause_list3],
            format_data=1,
            db_path='tl_daily'
        )
        results.rename(columns={'date': 'trade_date', 'Code': 'code'}, inplace=True)
        return results

    def fetch_futures_daily(self, begin_date, end_date, fut_code):
        """获取期货日线行情持仓数据 (持仓量变化作为主力持仓异动代理)"""
        name = 'market_fut'
        names = DBAPI.CustomizeFactory(self.kd_engine).name(name=name)
        clause_list = [
            names.trade_date >= begin_date,
            names.trade_date <= end_date,
            names.contractObject == fut_code
        ]
        daily_market = DBAPI.CustomizeFactory(self.kd_engine).custom(
            name=name, clause_list=clause_list, columns=['trade_date', 'code', 'openInt', 'turnoverVol'])
        if daily_market.empty:
            return pd.DataFrame()
            
        daily_market.rename(columns={'openInt': 'openint', 'turnoverVol': 'volume'}, inplace=True)
        # 聚合该品种所有合约的总持仓
        agg_fut = daily_market.groupby('trade_date')[['openint', 'volume']].sum().reset_index()
        return agg_fut

    def _create_smart_money_flow(self, comp_wide_flow):
        """
        1. 主力资金边际动能 (Smart Money Flow Acceleration) - 宽表矢量化计算
        计算成分股整体主力净流入额（mainFlow = 大单+特大单）的 5 日累积值及其二阶导（加速度）。
        """
        if comp_wide_flow.empty or 'mainFlow' not in comp_wide_flow:
            return pd.DataFrame()
            
        main_flow_wide = comp_wide_flow['mainFlow']
        turnover_wide = comp_wide_flow['turnoverValue']
        
        # 宽表跨股票横截面求和 (axis=1)
        daily_main_flow = main_flow_wide.sum(axis=1)
        daily_turnover = turnover_wide.sum(axis=1)
        
        # 主力净流入占成交额比例
        main_flow_ratio = daily_main_flow / (daily_turnover + 1e-9)
        
        # 5日主力净流入累积
        main_flow_5d = main_flow_ratio.rolling(5, min_periods=2).sum()
        
        # 二阶导（加速度）：5日动量的一阶差分
        main_flow_acc = main_flow_5d.diff(3)
        
        return pd.concat([main_flow_ratio, main_flow_5d, main_flow_acc],
                         axis=1,
                         keys=['main_flow_ratio', 'main_flow_5d_cum', 'main_flow_acceleration'])

    def _create_smart_position_change(self, fut_data):
        """
        2. 主力持仓突变异动 (Smart Position Shift)
        期货端总持仓量 3 日突变率与放量异动
        """
        if fut_data.empty:
            return pd.DataFrame()
            
        fut_df = fut_data.set_index('trade_date')
        oi_change_3 = fut_df['openint'].pct_change(3)
        oi_vol_ratio = fut_df['openint'] / (fut_df['volume'] + 1e-9)
        
        return pd.concat([oi_change_3, oi_vol_ratio],
                         axis=1,
                         keys=['oi_change_3d', 'oi_volume_ratio'])

    def start(self, begin_date, end_date):
        comp_df = self.fetch_index_components(begin_date, end_date)
        if not comp_df.empty:
            codes = comp_df['code'].unique().tolist()
            flow_df = self.fetch_moneyflow(begin_date, end_date, codes)
            valid_flow = pd.merge(comp_df, flow_df, on=['trade_date', 'code'], how='inner')
            # 宽表化: trade_date 为 Index, code 为 Columns
            comp_wide_flow = valid_flow.set_index(['trade_date', 'code']).unstack()
        else:
            comp_wide_flow = pd.DataFrame()
            
        fut_code = S2F_MAPPING.get(self.code, '')
        fut_df = self.fetch_futures_daily(begin_date, end_date, fut_code) if fut_code else pd.DataFrame()

        smart_flow_data = self._create_smart_money_flow(comp_wide_flow)
        position_data = self._create_smart_position_change(fut_df)

        final_features = pd.concat([smart_flow_data, position_data], axis=1)
        return final_features
