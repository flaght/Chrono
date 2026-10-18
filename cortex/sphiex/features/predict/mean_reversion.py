import pdb
import pandas as pd
import numpy as np
from jdw import DBAPI
from alphacopilot.api.data import DDBAPI, ddb_tools

F2S_MAPPING = {'IF': '000300', 'IH': '000016', 'IC': '000905', 'IM': '000852'}
S2F_MAPPING = {'000300': 'IF', '000016': 'IH', '000905': 'IC', '000852': 'IM'}


class MeanReversion(object):
    """
    均值回归与拐点反转类预测特征 (Mean Reversion & Turning Points Alphas)
    核心逻辑：买卖盘力竭、超买超卖、恐慌出清（逆势抄底/逃顶）。
    适合捕捉 T+1 ~ T+2 的左侧反弹与见顶预警。
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
        clause_list1 = ddb_tools.to_format('date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))
        clause_list2 = ddb_tools.to_format('date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))
        clause_list3 = ddb_tools.to_format('Code', 'in', codes)
        results = self.cusomize_api.custom(
            table='market_stock',
            columns=['date', 'Code', 'closePrice', 'lowestPrice', 'highestPrice', 'turnoverValue'],
            clause_list=[clause_list1, clause_list2, clause_list3],
            format_data=1,
            db_path='tl_daily'
        )
        results.rename(
            columns={
                'date': 'trade_date',
                'Code': 'code',
                'closePrice': 'close',
                'lowestPrice': 'low',
                'highestPrice': 'high',
                'turnoverValue': 'value'
            },
            inplace=True)
        return results

    def _create_breadth_divergence(self, comp_wide_data, index_data):
        """
        1. 市场宽度与背离度特征 (Market Breadth & Divergence Metrics) - 纯宽表矢量化计算
        :param comp_wide_data: 包含成分股各字段的 MultiIndex 宽表 (trade_date 为行, code 为列)
        :param index_data: 目标指数日线数据 DataFrame
        """
        if comp_wide_data.empty or 'close' not in comp_wide_data:
            return pd.DataFrame()
            
        close_wide = comp_wide_data['close']
        low_wide = comp_wide_data['low'] if 'low' in comp_wide_data else close_wide
        high_wide = comp_wide_data['high'] if 'high' in comp_wide_data else close_wide
        
        # -------------------------------------------------------------
        # 方案 A：成分股跌破 20 日均线比例 (comp_below_ma20_ratio)
        # 宽表矩阵直接计算 rolling(20).mean()，然后 axis=1 横截面求均值
        # -------------------------------------------------------------
        ma20_wide = close_wide.rolling(window=20, min_periods=5).mean()
        is_below_ma20_wide = (close_wide < ma20_wide).astype(float)
        comp_below_ma20_ratio = is_below_ma20_wide.mean(axis=1)
        
        # -------------------------------------------------------------
        # 方案 B：成分股 20 日高低区间相对位置的截面平均值 (comp_breadth_pos_20)
        # 宽表矩阵直接计算滚动高低点，无循环/无 groupby
        # -------------------------------------------------------------
        low20_wide = low_wide.rolling(window=20, min_periods=5).min()
        high20_wide = high_wide.rolling(window=20, min_periods=5).max()
        stock_pos_20_wide = (close_wide - low20_wide) / (high20_wide - low20_wide + 1e-9)
        comp_breadth_pos_20 = stock_pos_20_wide.mean(axis=1)
        
        # -------------------------------------------------------------
        # 指数自身 20 日相对位置与背离强度计算
        # -------------------------------------------------------------
        idx_df = index_data.drop_duplicates('trade_date').set_index('trade_date').sort_index()
        idx_close = idx_df['close']
        idx_low_20 = idx_close.rolling(20, min_periods=5).min()
        idx_high_20 = idx_close.rolling(20, min_periods=5).max()
        idx_pos_20 = (idx_close - idx_low_20) / (idx_high_20 - idx_low_20 + 1e-9)
        
        # 方案 B 连续背离强度：成分股平均位置领先回升，而指数仍处于低位时的底背离强度
        breadth_div_intensity_pos = comp_breadth_pos_20 - idx_pos_20
        
        # 方案 A 连续背离强度：指数处于低位但跌破均线的股票占比并未扩大的底背离强度
        breadth_div_intensity_ma = (1.0 - idx_pos_20) - comp_below_ma20_ratio
        
        res_df = pd.concat([comp_below_ma20_ratio, comp_breadth_pos_20, breadth_div_intensity_pos, breadth_div_intensity_ma],
                           axis=1,
                           keys=['comp_below_ma20_ratio', 'comp_breadth_pos_20', 'breadth_div_intensity_pos', 'breadth_div_intensity_ma'])
        return res_df

    def _create_turnover_exhaustion(self, index_data):
        """
        2. 换手率高潮衰竭度 (Turnover Exhaustion Ratio)
        输出连续值特征：
        - turnover_percentile_252: 成交额/换手率滚动分位数 [0, 1]
        - turnover_change_rate: 相对前一日的成交额变化率
        - exhaustion_intensity: 高位缩量衰竭连续强度
        """
        idx_df = index_data.drop_duplicates('trade_date').set_index('trade_date').sort_index()
        val = idx_df['value']
        
        val_pct_252 = val.rolling(window=252, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
        val_change = val.pct_change()
        exhaustion_intensity = val_pct_252.shift(1) * (-val_change)
        
        return pd.concat([val_pct_252, val_change, exhaustion_intensity],
                         axis=1,
                         keys=['turnover_percentile_252', 'turnover_change_rate', 'exhaustion_intensity'])

    def _create_liquidity_panic(self, index_data):
        """
        3. 极端流动性恐慌/下影线反转度 (Liquidity Panic & Reversal Ratio)
        输出连续值特征：
        - amplitude_percentile_252: 振幅滚动分位数 [0, 1]
        - lower_shadow_ratio: 下影线占全天振幅比例 [0, 1]
        - panic_reversal_score: 恐慌反弹连续打分 (振幅分位 * 下影线比例)
        """
        idx_df = index_data.drop_duplicates('trade_date').set_index('trade_date').sort_index()
        amplitude = (idx_df['high'] - idx_df['low']) / (idx_df['close'].shift(1) + 1e-9)
        lower_shadow = (np.minimum(idx_df['open'], idx_df['close']) - idx_df['low']) / (idx_df['high'] - idx_df['low'] + 1e-9)
        
        amp_pct_252 = amplitude.rolling(window=252, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            
        panic_reversal_score = amp_pct_252 * lower_shadow
        
        return pd.concat([amp_pct_252, lower_shadow, panic_reversal_score],
                         axis=1,
                         keys=['amplitude_percentile_252', 'lower_shadow_ratio', 'panic_reversal_score'])

    def start(self, begin_date, end_date):
        index_data = self.fetch_spot_daily(begin_date, end_date, self.code)
        
        comp_df = self.fetch_index_components(begin_date, end_date)
        if not comp_df.empty:
            codes = comp_df['code'].unique().tolist()
            market_df = self.fetch_stock_market(begin_date, end_date, codes)
            valid_comp_market = pd.merge(comp_df, market_df, on=['trade_date', 'code'], how='inner')
            # 转为标准的宽表格式: trade_date 为 Index, code 为 Columns (与 price_feature 规范完全一致)
            comp_wide_data = valid_comp_market.set_index(['trade_date', 'code']).unstack()
        else:
            comp_wide_data = pd.DataFrame()

        breadth_div_data = self._create_breadth_divergence(comp_wide_data, index_data)
        turnover_data = self._create_turnover_exhaustion(index_data)
        panic_data = self._create_liquidity_panic(index_data)

        final_features = pd.concat([breadth_div_data, turnover_data, panic_data], axis=1)
        return final_features
