import pdb
import pandas as pd
import numpy as np
from jdw import DBAPI
from ultron.tradingday import *
from alphacopilot.api.data import DDBAPI, ddb_tools

MAPPING_CODE = {'IF': '510300', 'IH': '510050', 'IC': '510500', 'IM': '510300'}


class VolatilityFeature(object):

    def __init__(self, code, engine=None):
        self.kd_engine = DBAPI.FetchEngine.create_engine(
            'kd') if engine is None else engine
        self.cusomize_api = DDBAPI.cusomize_api()
        self.code = code

    def fetch_opt_hv(self, begin_date, end_date):
        clause_list1 = ddb_tools.to_format('Code', 'in',
                                           [MAPPING_CODE[self.code]])
        clause_list2 = ddb_tools.to_format(
            'date', '<=', ddb_tools.convert_date(end_date.replace('-', '.')))

        clause_list3 = ddb_tools.to_format(
            'date', '>=', ddb_tools.convert_date(begin_date.replace('-', '.')))

        clause_list4 = ddb_tools.to_format('period', 'in', [10, 60])
        results = self.cusomize_api.custom(
            table='opt_ul_hv',
            columns=['date', 'Code', 'period', 'HV'],
            clause_list=[
                clause_list1, clause_list2, clause_list3, clause_list4
            ],
            format_data=1,
            db_path='tl_daily')
        return results

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

    # def load_data(self, begin_date, end_date):
    #     opt_hv_data = self.fetch_opt_hv(begin_date=begin_date,
    #                                     end_date=end_date)

    #     return opt_hv_data.rename(columns={
    #         'date': 'trade_date',
    #         'Code': 'code'
    #     })

    def load_data(self, begin_date, end_date):
        market_data = self.fetch_spot_daily(begin_date=begin_date,
                                            end_date=end_date,
                                            code=self.code)

        return market_data

    def _create_opt_hv(self, base_data):
        wide_hv = base_data.pivot(index='trade_date',
                                  columns='period',
                                  values='HV').reset_index()
        rename_dict = {
            10: 'short_hv',
            60: 'mid_hv',
        }
        wide_hv = wide_hv.rename(columns=rename_dict)

        if 'short_hv' not in wide_hv.columns:
            wide_hv['short_hv'] = float('nan')
        if 'mid_hv' not in wide_hv.columns:
            wide_hv['mid_hv'] = float('nan')

        wide_hv['code'] = self.code
        final_df = wide_hv[['trade_date', 'code', 'short_hv', 'mid_hv']]
        final_df.columns.name = None
        return final_df

    def _create_realized_volatility(self, base_data):
        """9. 历史实现波动率 (Realized Volatility - 自主计算替代期权HV)"""
        returns = base_data['close'].pct_change()
        annualization_factor = np.sqrt(252)

        rv_5 = returns.rolling(window=5).std() * annualization_factor
        rv_25 = returns.rolling(window=25).std() * annualization_factor
        rv_75 = returns.rolling(window=75).std() * annualization_factor
        rv_250 = returns.rolling(window=250).std() * annualization_factor

        return pd.concat([rv_5, rv_25, rv_75, rv_250],
                         axis=1,
                         keys=['rv_5', 'rv_25', 'rv_75', 'rv_250'])

    def start(self, begin_date, end_date):
        base_data = self.load_data(begin_date=begin_date, end_date=end_date)
        wide_data = base_data.set_index(['trade_date', 'code']).unstack()
        # opt_hv_data = self._create_opt_hv(base_data)
        rv_data = self._create_realized_volatility(base_data=wide_data)
        return rv_data.stack()
