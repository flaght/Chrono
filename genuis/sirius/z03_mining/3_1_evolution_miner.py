from pathlib import Path
import pdb

import polars as pl

from lib.data.loader.data1 import load_basis_lazy, load_returns
from z03_mining.miner.evolution import EvolutionConfig, Launcher

period = 5
return_name = f"spot_ret_{period}h"

symbol = "BTCUSDT"
futures_dir = f"z21_orchestrator/data/{symbol}/basic/futures"
spot_dir = f"z21_orchestrator/data/{symbol}/basic/spot"
returns_file = f"z21_orchestrator/data/{symbol}/returns/spot.feather"
output_dir = f"z21_orchestrator/results/{period}"

if __name__ == "__main__":
    basis_lazy = load_basis_lazy(
        futures_dir= futures_dir,
        spot_dir=spot_dir,
        symbol=symbol,
    )
    pdb.set_trace()
    returns_lazy = load_returns(returns_file)


    combined_lazy = (
        basis_lazy
        .join(returns_lazy, on=["trade_time", "code"], how="inner")
        .sort(["trade_time", "code"])  # 保证因子挖掘时时间序列连续有序
    )


    config = EvolutionConfig(
        population_size=30,
        generations=5,
        tournament_size=5,
        elite_size=5,
        max_depth=5,
        periods=(period,),
    )

    
    features = [col for col in basis_lazy.collect_schema().names() if col not in ["trade_time", "code"]]
    return_column  = f"spot_ret_{period}h"
    miner = Launcher(
        feature_names=features,
        return_column=return_column,
        mode='free',
        config=config,
        operator_config= None,
    )
    pdb.set_trace()
    result = miner.optimize(combined_lazy, top_n=10)
    if True:
        output = Path('./temp')
        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_ipc(output)
    print(result)