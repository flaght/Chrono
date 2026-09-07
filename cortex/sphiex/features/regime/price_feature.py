import pdb
import pandas as pd
import numpy as np
from jdw import DBAPI
from alphacopilot.api.data import DDBAPI, ddb_tools

F2S_MAPPING = {'IF': '000300', 'IH': '000016', 'IC': '000905', 'IM': '000852'}
S2F_MAPPING = {'000300': 'IF', '000016': 'IH', '000905': 'IC', '000852': 'IM'}


## 预测指数
class PriceFeature(object):

    def __init__(self, code, engine=None):
        self.kd_engine = DBAPI.FetchEngine.create_engine(
            'kd') if engine is None else engine
        self.cusomize_api = DDBAPI.cusomize_api()
        self.code = code

    def fetch_daily(self, begin_date, end_date, codes, columns=None):
        name = 'market_fut'
        names = DBAPI.CustomizeFactory(self.kd_engine).name(name=name)
        clause_list = [
            names.trade_date >= begin_date,
            names.trade_date <= end_date,
        ]
        if isinstance(codes, list):
            clause_list.append(names.code.in_(codes))
        daily_market = DBAPI.CustomizeFactory(self.kd_engine).custom(
            name=name, clause_list=clause_list, columns=columns)
        daily_market.rename(columns={
            'code': 'symbol',
            'contractObject': 'code',
            'openPrice': 'open',
            'highestPrice': 'high',
            'lowestPrice': 'low',
            'closePrice': 'close',
            'turnoverVol': 'volume',
            'turnoverValue': 'value',
            'openInt': 'openint'
        },
                            inplace=True)
        return daily_market[[
            'trade_date', 'code', 'symbol', 'open', 'high', 'low', 'close',
            'volume', 'value', 'openint'
        ]]

    def fetch_algin_factors(self,
                            begin_date,
                            end_date,
                            codes=None,
                            columns=None):
        name = 'fut_algin_factors'
        names = DBAPI.CustomizeFactory(self.kd_engine).name(name=name)
        clause_list = [
            names.trade_date >= begin_date, names.trade_date <= end_date
        ]
        if isinstance(codes, list):
            clause_list.append(names.code.in_(codes))
        algin_factors_data = DBAPI.CustomizeFactory(self.kd_engine).custom(
            name=name, clause_list=clause_list, columns=columns)
        return algin_factors_data

    def fetch_feature_daily(self, begin_date, end_date, code):
        adj_name = "{0}_cumfactor".format("pcr")
        algin_factors = self.fetch_algin_factors(
            begin_date=begin_date,
            end_date=end_date,
            codes=[code],
            columns=['trade_date', 'code', 'symbol', adj_name])
        algin_factors = (algin_factors.drop_duplicates(
            subset=["trade_date", "code"],
            keep="last").sort_values(["code",
                                      "trade_date"]).reset_index(drop=True))
        change = (algin_factors["symbol"].ne(
            algin_factors.groupby("code")["symbol"].shift())
                  | algin_factors["pcr_cumfactor"].ne(
                      algin_factors.groupby("code")["pcr_cumfactor"].shift()))
        algin_factors["grp"] = (change.groupby(algin_factors["code"]).cumsum())

        ranges = (algin_factors.groupby(["code", "grp"], as_index=False).agg(
            symbol=("symbol", "first"),
            start_date=("trade_date", "min"),
            end_date=("trade_date", "max"),
            pcr_cumfactor=("pcr_cumfactor", "first"),
        ))
        res = []
        adj_columns = ['open', 'high', 'low', 'close']
        for row in ranges.itertuples():
            md = self.fetch_daily(
                begin_date=row.start_date.strftime('%Y-%m-%d'),
                end_date=row.end_date.strftime('%Y-%m-%d'),
                codes=[row.symbol],
                columns=None)
            md['factor'] = row.pcr_cumfactor
            md[adj_columns] = md[adj_columns].multiply(md['factor'], axis=0)
            res.append(md)
        market_data = pd.concat(res,
                                axis=0).sort_values(by=['trade_date', 'code'])
        return market_data

    def fetch_spot_daily(self, begin_date, end_date, code):
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
        f_data = self.fetch_feature_daily(begin_date=begin_date,
                                          end_date=end_date,
                                          code=S2F_MAPPING[self.code])
        s_data = self.fetch_spot_daily(begin_date=begin_date,
                                       end_date=end_date,
                                       code=self.code)
        # market_data = f_data[[
        #     'trade_date', 'code', 'open', 'close', 'high', 'low', 'volume'
        # ]].merge(s_data[['trade_date',
        #                  'close']].rename(columns={'close': 'spot_close'}),
        #          on=['trade_date'])
        s_data['spot_close'] = s_data['close'].copy()
        market_data = s_data[[
            'trade_date', 'code', 'open', 'close', 'high', 'low', 'volume',
            'spot_close'
        ]].merge(f_data[[
            'trade_date',
            'close',
        ]].rename(columns={'close': 'feature_close'}),
                 on=['trade_date'],
                 how='left')
        return market_data

    def _create_trend_ma(self, base_data):
        """1. 均线排列状态"""
        # 此时 base_data['close'] 是一个包含了(IM, IC, IF, IH)各列的 DataFrame，下同
        ma5 = base_data['close'].rolling(window=5).mean()
        ma20 = base_data['close'].rolling(window=20).mean()
        ma60 = base_data['close'].rolling(window=60).mean()

        # 利用 pd.concat 加上 keys，拼接出完美的 MultiIndex 列名结构
        return pd.concat([ma5, ma20, ma60],
                         axis=1,
                         keys=['ma5', 'ma20', 'ma60'])

    def _create_price_position(self, base_data):
        """2. 价格箱体位置"""
        high_60 = base_data['high'].rolling(window=60).max()
        low_60 = base_data['low'].rolling(window=60).min()
        position_box = (base_data['close'] - low_60) / (high_60 - low_60 +
                                                        1e-9)
        return pd.concat([high_60, low_60, position_box],
                         axis=1,
                         keys=['high_60', 'low_60', 'position_box'])

    def _create_volatility_atr(self, base_data):
        """3. ATR波动率分位"""
        prev_close = base_data['close'].shift(1)

        tr1 = base_data['high'] - base_data['low']
        tr2 = (base_data['high'] - prev_close).abs()
        tr3 = (base_data['low'] - prev_close).abs()

        # 宽表级别求 Max，必须使用 np.maximum 进行二维数组的逐元素对比
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr_14 = tr.rolling(window=14).mean()
        atr_pct_252 = atr_14.rolling(window=252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        return pd.concat([atr_14, atr_pct_252],
                         axis=1,
                         keys=['atr_14', 'atr_pct_252'])

    def _create_drawdown(self, base_data):
        """4. 中期回撤幅度"""
        high_60 = base_data['high'].rolling(window=60).max()
        drawdown_60 = (base_data['close'] / high_60) - 1
        return pd.concat([drawdown_60], axis=1, keys=['drawdown_60'])

    def _create_volume_ratio(self, base_data):
        """5. 相对成交量倍数"""
        vol_ma20 = base_data['volume'].shift(1).rolling(window=20).mean()
        volume_ratio = base_data['volume'] / (vol_ma20 + 1e-9)
        return pd.concat([volume_ratio], axis=1, keys=['volume_ratio'])

    def _create_gap(self, base_data):
        """6. 跳空缺口状态"""
        # 注意：这里需要您的原数据里有 open 列
        gap_pct = (base_data['open'] / base_data['close'].shift(1)) - 1
        return pd.concat([gap_pct], axis=1, keys=['gap_pct'])

    def _create_basis(self, base_data):
        """8. 基差状态与升贴水"""
        basis = base_data['feature_close'] - base_data['spot_close']
        basis_pct = basis / base_data['spot_close']
        basis_percentile_252 = basis_pct.rolling(
            window=252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        return pd.concat([basis_pct, basis_percentile_252],
                         axis=1,
                         keys=['basis_pct', 'basis_percentile_252'])

    def _create_crowdedness_turnover(self, base_data):
        """10. 换手率拥挤度状态 (使用现货成交额分位数作为代理)"""
        if 'spot_value' not in base_data.columns.levels[0]:
            return pd.DataFrame()

        turnover_val = base_data['spot_value']
        turnover_percentile_252 = turnover_val.rolling(
            window=252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        return pd.concat([turnover_percentile_252],
                         axis=1,
                         keys=['turnover_percentile_252'])

    def _create_return_distribution(self, base_data):
        """11. 收益率分布特征状态 (20日收益率峰度与偏度)"""
        returns = base_data['spot_close'].pct_change()
        skew_20 = returns.rolling(window=20).skew()
        kurt_20 = returns.rolling(window=20).kurt()
        return pd.concat([skew_20, kurt_20],
                         axis=1,
                         keys=['return_skew_20', 'return_kurt_20'])

    def _create_candle_shape(self, base_data):
        """12. 最近K线形态"""
        amplitude = base_data['high'] - base_data['low']
        # 避免除以0
        amplitude = amplitude.replace(0, np.nan)
        body_ratio = (base_data['close'] - base_data['open']).abs() / amplitude
        upper_shadow_ratio = (base_data['high'] - np.maximum(base_data['open'], base_data['close'])) / amplitude
        lower_shadow_ratio = (np.minimum(base_data['open'], base_data['close']) - base_data['low']) / amplitude
        
        return pd.concat([body_ratio, upper_shadow_ratio, lower_shadow_ratio], 
                         axis=1, 
                         keys=['body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio'])

    def _create_price_return(self, base_data):
        """13. 价格收益率"""
        return_5 = base_data['close'].pct_change(periods=5)
        return_20 = base_data['close'].pct_change(periods=20)
        return pd.concat([return_5, return_20],
                         axis=1,
                         keys=['return_5', 'return_20'])

    def _create_oi_change(self, base_data):
        """14. 持仓量变化"""
        if 'openint' not in base_data.columns.levels[0]:
            return pd.DataFrame()
        oi_change_5 = base_data['openint'].pct_change(periods=5)
        oi_change_20 = base_data['openint'].pct_change(periods=20)
        return pd.concat([oi_change_5, oi_change_20],
                         axis=1,
                         keys=['oi_change_5', 'oi_change_20'])

    def start(self, begin_date, end_date):
        market_data = self.load_data(begin_date=begin_date, end_date=end_date)
        wide_data = market_data.set_index(['trade_date', 'code']).unstack()

        trend_ma_data = self._create_trend_ma(base_data=wide_data)
        price_position_data = self._create_price_position(base_data=wide_data)
        volatility_atr_data = self._create_volatility_atr(base_data=wide_data)
        drawdown_data = self._create_drawdown(base_data=wide_data)
        volume_ratio_data = self._create_volume_ratio(base_data=wide_data)
        gap_data = self._create_gap(base_data=wide_data)
        basis_data = self._create_basis(base_data=wide_data)
        crowdedness_data = self._create_crowdedness_turnover(
            base_data=wide_data)
        return_dist_data = self._create_return_distribution(
            base_data=wide_data)
        candle_shape_data = self._create_candle_shape(base_data=wide_data)
        price_return_data = self._create_price_return(base_data=wide_data)
        oi_change_data = self._create_oi_change(base_data=wide_data)
        feature_dfs = [
            trend_ma_data, price_position_data, volatility_atr_data,
            drawdown_data, volume_ratio_data, gap_data, basis_data,
            crowdedness_data, return_dist_data, candle_shape_data,
            price_return_data, oi_change_data
        ]
        final_features = pd.concat(feature_dfs, axis=1)
        return final_features
