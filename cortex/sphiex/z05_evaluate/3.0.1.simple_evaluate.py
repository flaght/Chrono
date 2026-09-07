import json, os, asyncio, pdb
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from kdutils.macro import base_path


def load_results(method, period):
    dir_path = Path(os.path.join(base_path, "outofsam", method, str(period)))
    snapshot_dict = {}
    for json_file in dir_path.glob("*.json"):
        date_str = json_file.stem.split("_")[0]
        with open(json_file, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
            snapshot_dict[date_str] = snapshot
    return snapshot_dict


async def run(method, period, lookback):
    ticker = "000852"
    snapshot_dict = await asyncio.to_thread(load_results,
                                            method=method,
                                            period=period)
    res = []
    for k, v in snapshot_dict.items():
        predict_direction = v['trader_prediction']['predict_direction']
        if predict_direction == "UP":
            direction= 1
        elif predict_direction == "DOWN":
            direction = -1
        elif predict_direction == "FLAT":
            direction = 0
        confidence = v['trader_prediction']['confidence']
        forward_return = v['forward_return']
        res.append({
            'trade_date':k,
            'direction': direction,
            'confidence': confidence,
            'forward_return': forward_return
        })
    results = pd.DataFrame(res).sort_values(by=['trade_date']).reset_index(drop=True)
    results['returns1'] = (results['direction'] * results['forward_return'] / 3)
    pdb.set_trace()
    print('-->')


if __name__ == '__main__':
    method = 'train0'
    period = 3
    asyncio.run(run(method=method, period=3, lookback=3))
