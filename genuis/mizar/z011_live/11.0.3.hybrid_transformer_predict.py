import re
import pandas as pd
from pathlib import Path
from lib.nnl005.workflow import *
from lib.nnl005.workflow import Workflow as LiveWorkFlow


def read_bars(path: str, code: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".feather", ".arrow"):
        frame = pd.read_feather(path, columns=list(BAR_COLUMNS))
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, usecols=list(BAR_COLUMNS))

    frame = frame.loc[frame["code"].astype(str).str.upper() == code,
                      list(BAR_COLUMNS)].copy()
    if frame.empty:
        raise ValueError(f"{path} 中没有 {code} 的 bar")
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="raise")
    return frame.sort_values("trade_time").reset_index(drop=True)


def batch(code, max_bars=None, batch_size=256):
    #bars_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/basic/test_data.feather'
    #release_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/1release/rbb/10002'
    #directions_file = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/rulex/113001/nxt1_ret_5h/cohort_pro.csv'
    # signals_method = [{
    #     'signal_method': 'threshold_signal',
    #     'signal_params': {
    #         "roll_num": 0,
    #         "threshold": 0.02
    #     }
    # }, {
    #     'signal_method': 'linear_signal',
    #     'signal_params': {
    #         "roll_num": 0,
    #         "threshold": 0.15
    #     },
    # }]
    bars_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/pc/basic/test_data.feather'
    release_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/1release/pc/10001'
    directions_file = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/pc/rulex/159001/nxt1_ret_5h/cohort_pro.csv'
    signals_method = [{
        'signal_method': 'linear_signal',
        'signal_params': {
            'roll_num': 0,
            'threshold': 0.05
        }
    }]

    device = 'cpu'
    history_bars = 1500
    min_raw_bars = 300
    target = read_bars(path=bars_path, code=code)
    if max_bars is not None:
        target = target.head(max_bars).copy()
    live = LiveWorkFlow(release_path=release_path,
                        directions_file=directions_file,
                        code=code,
                        device=device,
                        history_bars=history_bars,
                        min_raw_bars=min_raw_bars,
                        signals_method=signals_method)

    return live.predict_batch(target, batch_size=batch_size)


def live(code, max_bars=None) -> None:
    bars_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/basic/test_data.feather'
    release_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/1release/rbb/10002'
    directions_file = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/rulex/113001/nxt1_ret_5h/cohort_pro.csv'
    device = 'cpu'
    history_bars = 1500
    min_raw_bars = 300
    signals_method = [{
        'signal_method': 'threshold_signal',
        'signal_params': {
            "roll_num": 0,
            "threshold": 0.02
        }
    }, {
        'signal_method': 'linear_signal',
        'signal_params': {
            "roll_num": 0,
            "threshold": 0.15
        },
    }]
    target = read_bars(path=bars_path, code=code)
    if max_bars is not None:
        target = target.head(max_bars).copy()
    target = target.head(max_bars).copy()

    combined = target.copy()
    live = LiveWorkFlow(release_path=release_path,
                        directions_file=directions_file,
                        code=code,
                        device=device,
                        history_bars=history_bars,
                        min_raw_bars=min_raw_bars,
                        signals_method=signals_method)

    res = []
    for index, row in enumerate(combined.itertuples(index=False, name=None)):
        result = live.on_bar(dict(zip(BAR_COLUMNS, row)))
        res.append(result)
    result = pd.concat(res, axis=0)


#live(code='RB', max_bars=1500)
batch(code='P', max_bars=None)
