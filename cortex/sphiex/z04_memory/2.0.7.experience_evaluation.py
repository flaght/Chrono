import asyncio, os, pdb, json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro import base_path
from lib.amr001 import MARLMemoryCoordinator


def load_results(method, period):
    dir_path = Path(os.path.join(base_path, "enhanced", method, str(period)))
    snapshot_dict = {}
    for json_file in dir_path.glob("*.json"):
        date_str = json_file.stem.split("_")[0]
        with open(json_file, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
            snapshot_dict[date_str] = snapshot
    return snapshot_dict


def load_data(method, period):
    ### 需要进行标准化处理
    predict_data = pd.read_feather(
        os.path.join("records", "normal", str(method), "predict_data.feather"))
    regime_data = pd.read_feather(
        os.path.join("records", "normal", str(method), "regime_data.feather"))
    textuals_data = pd.read_feather(
        os.path.join("records", "normal", str(method),
                     "textuals_data.feather"))
    returns_data = pd.read_feather(
        os.path.join("records", "normal", str(method), "returns_data.feather"))
    returns_data = returns_data[[
        'trade_date', 'code', "nxt1_ret_{0}h".format(period)
    ]]
    predict_data['trade_date'] = pd.to_datetime(predict_data['trade_date'])
    regime_data['trade_date'] = pd.to_datetime(regime_data['trade_date'])
    textuals_data['trade_date'] = pd.to_datetime(textuals_data['trade_date'])
    return predict_data, regime_data, textuals_data, returns_data


async def run(method, period, lookback):
    ticker = "000852"
    snapshot_dict = await asyncio.to_thread(load_results,
                                            method=method,
                                            period=period)
    predict_data, regime_data, textuals_data, returns_data = await asyncio.to_thread(
        load_data, method=method, period=period)

    pdb.set_trace()
    p_cols = [
        f for f in predict_data.columns if not f in ['trade_date', 'code']
    ]
    r_cols = [
        f for f in regime_data.columns if not f in ['trade_date', 'code']
    ]
    p_dim = len(p_cols) * (lookback + 1)
    r_dim = len(r_cols) * (lookback + 1)
    
    storage_path = os.path.join(base_path, "brain", method, str(period))
    coordinator = MARLMemoryCoordinator(name="ashare_{0}".format(ticker),
                                        storage_path=storage_path,
                                        vector_provider='fassis',
                                        embedding_model='text-embedding-v4',
                                        embedding_provider='openai',
                                        p_dim=p_dim,
                                        r_dim=r_dim)
    
    for k, v in snapshot_dict.items():
        coordinator.update_memory(
            sample_id=v['sample_id'],
            symbol=ticker,
            trader_prediction=v['trader_prediction'],
            realized_return=snapshot_dict['2026-07-27']['forward_return'])


if __name__ == '__main__':
    method = 'test0'
    period = 3
    asyncio.run(run(method=method, period=3, lookback=3))
