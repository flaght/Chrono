import pdb
import pandas as pd
import numpy as np
from jdw import DBAPI
from alphacopilot.api.data import DDBAPI, ddb_tools

F2S_MAPPING = {'IF': '000300', 'IH': '000016', 'IC': '000905', 'IM': '000852'}
S2F_MAPPING = {'000300': 'IF', '000016': 'IH', '000905': 'IC', '000852': 'IM'}


class TrendMomentum(object):
    """
    趋势与动量追踪类预测特征 (Trend & Momentum Alphas)
    核心逻辑：强者恒强、资金惯性、突破确认（顺势买入/卖出）。
    适合捕捉 T+1 ~ T+2 的单边惯性溢价。
    """

    def __init__(self, code, engine=None):
        self.kd_engine = DBAPI.FetchEngine.create_engine(
            'kd') if engine is None else engine
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
        results.rename(columns={
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

    def load_data(self, begin_date, end_date):
        """加载目标标的及基准(000300)行情"""
        target_data = self.fetch_spot_daily(begin_date, end_date, self.code)
        hs300_data = self.fetch_spot_daily(begin_date, end_date, '000300')

        combined = pd.concat(
            [target_data, hs300_data],
            axis=0).drop_duplicates(subset=['trade_date', 'code'])
        return combined

    def _create_short_term_momentum(self, base_data):
        """
        1. 截面微观动量 (Idiosyncratic Short-term Momentum)
        过去20日对数收益率之和，并结合滚动波动率进行时序动量标准化 (ret / sigma^2)
        """
        # base_data 为宽表: base_data['close'][self.code]
        close = base_data['close'][self.code]
        log_ret = np.log(close / close.shift(1))
        momentum_20 = log_ret.rolling(window=20).sum()
        vol_20 = log_ret.rolling(window=20).std()

        # 波动率调整后的动量
        momentum_adj = momentum_20 / (vol_20**2 + 1e-9)

        # 滚动252日分位数
        momentum_pct_252 = momentum_20.rolling(
            window=252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        return pd.concat(
            [momentum_20, momentum_adj, momentum_pct_252],
            axis=1,
            keys=['momentum_20', 'momentum_vol_adj_20', 'momentum_pct_252'])

    def _create_morning_gap_momentum(self, base_data):
        """
        2. 早盘跳空延续性信号 (Morning Gap Momentum)
        计算当日开盘相对昨日收盘的跳空幅度，以及结合日内实体强度的延续性
        """
        open_price = base_data['open'][self.code]
        close_price = base_data['close'][self.code]
        prev_close = close_price.shift(1)

        gap_pct = (open_price / prev_close) - 1
        intraday_ret = (close_price / open_price) - 1

        # 跳空且日内同向延续强度: gap * intraday_ret > 0 表示方向一致
        gap_continuation = gap_pct * np.sign(intraday_ret)

        return pd.concat([gap_pct, gap_continuation],
                         axis=1,
                         keys=['gap_pct', 'gap_continuation'])

    def _create_style_rotation_momentum(self, base_data):
        """
        3. 大小盘轮动剪刀差动能 (Style Rotation Momentum)
        (目标指数收益率 - 沪深300收益率) 的 5 日动量斜率
        """
        close = base_data['close']
        if self.code not in close.columns or '000300' not in close.columns:
            return pd.DataFrame()

        target_ret = close[self.code].pct_change(periods=5)
        hs300_ret = close['000300'].pct_change(periods=5)

        style_spread_5 = target_ret - hs300_ret
        style_spread_ma5 = style_spread_5.rolling(window=5).mean()
        style_momentum_slope = style_spread_5 - style_spread_ma5

        return pd.concat([style_spread_5, style_momentum_slope],
                         axis=1,
                         keys=['style_spread_5', 'style_momentum_slope'])

    def start(self, begin_date, end_date):
        market_data = self.load_data(begin_date=begin_date, end_date=end_date)
        wide_data = market_data.set_index(['trade_date', 'code']).unstack()
        
        mom_data = self._create_short_term_momentum(base_data=wide_data)
        gap_data = self._create_morning_gap_momentum(base_data=wide_data)
        style_data = self._create_style_rotation_momentum(base_data=wide_data)

        final_features = pd.concat([mom_data, gap_data, style_data], axis=1)
        return final_features
