import pdb,os
from dotenv import load_dotenv
load_dotenv()

from lib.utils.ttimes import get_dates
from lib.utils.macro import base_path
from lib.data.scan_data import scan_file
from feature.mc001.aggregator import aggregate_mc001
from feature.mc002.aggregator import aggregate_mc002
from feature.mc003.aggregator import aggregate_mc003
from feature.mc004.aggregator import aggregate_mc004
from feature.mf001.aggregator import aggregate_mf001
from feature.mf002.aggregator import aggregate_mf002 


def create_factors(method, instruments):
    start_date, end_date = get_dates(method)
    filename = os.path.join(base_path, method, instruments, 'basic', 'tick_data.feather')
    df_lazy = scan_file(filename)
    pdb.set_trace()
    mc001_data = aggregate_mc001(df_lazy)
    mc002_data = aggregate_mc002(df_lazy)
    mc003_data = aggregate_mc003(df_lazy)
    mc004_data = aggregate_mc004(df_lazy)
    mf001_data = aggregate_mf001(df_lazy)
    mf002_data = aggregate_mf002(df_lazy)

    output_path = os.path.join(base_path, method, instruments, 'factors')
    os.makedirs(output_path, exist_ok=True)
    pdb.set_trace()

    
    mc001_data.collect().to_pandas().to_feather(os.path.join(output_path, 'mc001_data.feather'))
    mc002_data.collect().to_pandas().to_feather(os.path.join(output_path, 'mc002_data.feather'))
    mc003_data.collect().to_pandas().to_feather(os.path.join(output_path, 'mc003_data.feather'))
    mc004_data.collect().to_pandas().to_feather(os.path.join(output_path, 'mc004_data.feather'))
    mf001_data.collect().to_pandas().to_feather(os.path.join(output_path, 'mf001_data.feather'))
    mf002_data.collect().to_pandas().to_feather(os.path.join(output_path, 'mf002_data.feather'))


if __name__ == '__main__':
    create_factors(method='ricso2', instruments='rbb')
