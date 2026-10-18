### dolphiDB 入库
import os, pdb
from kdutils.dolphindb1 import DolphinDBAdapter

data_client = DolphinDBAdapter(uri=os.environ['DDB_URL'])


def insert_factor_series1(series_data, factor_name, code):
    series_data1 = series_data.reset_index()
    series_data1['name'] = factor_name
    series_data1['Code'] = code
    series_data1['date'] = series_data1['trade_time'].dt.normalize()
    series_data1.rename(columns={factor_name: 'value'}, inplace=True)
    data_client.refresh_data(method='full',
                             table_name='scope_raw_factors',
                             db_name='miz_min',
                             total_data=series_data1)

def insert_metrics_data(df_data):
    df_data1 = df_data.copy()
    df_data1.rename(columns={'code':'Code'}, inplace=True)
    df_data1['date'] = df_data1['trade_time'].dt.normalize()
    data_client.refresh_data(method='full',
                             table_name='scope_factors_metrics',
                             db_name='miz_min',
                             total_data=df_data1)
    
def insert_returns_series1(series_data, code):
    series_data1 = series_data.reset_index()
    series_data1['Code'] = code
    series_data1['date'] = series_data1['trade_time'].dt.normalize()
    series_data1.rename(columns={'nxt1_ret': 'value'}, inplace=True)
    data_client.refresh_data(method='full',
                             table_name='scope_raw_returns',
                             db_name='miz_min',
                             total_data=series_data1)

def insert_elite_factor(factor_infos, code):
    pdb.set_trace()
    factor_infos['Code']=code
    data_client.refresh_data(method='full',
                             table_name='elite_factor_info',
                             db_name='miz_static',
                             total_data=factor_infos)