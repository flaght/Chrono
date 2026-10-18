
import pdb,os
from dotenv import load_dotenv
from z02_features.feature import tc005, tf001
load_dotenv()

from lib.data.loader.data0 import load_basis_lazy, load_data_lazy
from feature.tc001.aggregator import aggregate_tc001
from feature.tc002.aggregator import aggregate_tc002
from feature.tc003.aggregator import aggregate_tc003
from feature.tc004.aggregator import aggregate_tc004
from feature.tc005.aggregator import aggregate_tc005
from feature.tc006.aggregator import aggregate_tc006
from feature.tf001.aggregator import aggregate_tf001
from feature.tf002.aggregator import aggregate_tf002

period = 5
return_name = f"spot_ret_{period}h"

code = "BTCUSDT"
futures_dir = f"z21_orchestrator/data/{code}/basic/futures"
spot_dir = f"z21_orchestrator/data/{code}/basic/spot"
returns_file = f"z21_orchestrator/data/{code}/returns/spot.feather"
output_dir = f"z21_orchestrator/data/{code}/features"



def create_spot_factors():
    spot_lazy = load_data_lazy(
        data_dir=spot_dir,
        code=code
    )
    tc001_data = aggregate_tc001(spot_lazy)
    tc002_data = aggregate_tc002(spot_lazy)
    tc003_data = aggregate_tc003(spot_lazy)
    tc004_data = aggregate_tc004(spot_lazy)
    tc005_data = aggregate_tc005(spot_lazy)
    tc006_data = aggregate_tc006(spot_lazy)

    tc001_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc001_spot_data.feather'))
    tc002_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc002_spot_data.feather'))
    tc003_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc003_spot_data.feather'))
    tc004_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc004_spot_data.feather'))
    tc005_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc005_spot_data.feather'))
    tc006_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc006_spot_data.feather'))


def create_futures_factors():
    futures_lazy = load_data_lazy(
        data_dir=spot_dir,
        code=code
    )
    tc001_data = aggregate_tc001(futures_lazy)
    tc002_data = aggregate_tc002(futures_lazy)
    tc003_data = aggregate_tc003(futures_lazy)
    tc004_data = aggregate_tc004(futures_lazy)
    tc005_data = aggregate_tc005(futures_lazy)
    tc006_data = aggregate_tc006(futures_lazy)
    #tf001_data = aggregate_tf001(futures_lazy)
    #tf002_data = aggregate_tf002(futures_lazy)

    tc001_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc001_futures_data.feather'))
    tc002_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc002_futures_data.feather'))
    tc003_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc003_futures_data.feather'))
    tc004_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc004_futures_data.feather'))
    tc005_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc005_futures_data.feather'))
    tc006_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tc006_futures_data.feather'))
    #tf001_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tf001_futures_data.feather'))
    #tf002_data.collect().to_pandas().to_feather(os.path.join(output_dir, 'tf002_futures_data.feather'))


def create_factors():
    basis_lazy = load_basis_lazy(
        futures_dir= futures_dir,
        spot_dir=spot_dir,
        code=code,
    )


create_spot_factors()
create_futures_factors()
