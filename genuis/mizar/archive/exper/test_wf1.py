import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from lib.uvx import *
from kdutils.macro2 import *
from chaosmind.timing.sirius0003.workflow import WorkFlow
from chaosmind.timing.sirius0003.repository import Repository
from chaosmind.timing.sirius0003.repository import make_state_key
from chaosmind.timing.sirius0003.helpers import TradeOnline2
from kdutils.mongodb import MongoDBManager


def _sanitize_frame(df: pd.DataFrame, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    bad_mask = ~np.isfinite(df[cols].to_numpy(dtype=np.float64))
    bad_count = int(bad_mask.sum())
    if bad_count > 0:
        print(f"[WARN] 数据中发现 {bad_count} 个 NaN/Inf，已填充为 0.0")
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def load_data0(method, instruments, task_id, period, features, regime,
               ret_name, category):
    base_dirs = os.path.join(base_path, method, instruments, 'temp', 'model',
                             str(task_id), str(period), 'rl', 'data')
    if category == 'train':
        data = pd.read_feather(os.path.join(base_dirs, "train_data.feather"))
    elif category == 'val':
        data = pd.read_feather(os.path.join(base_dirs, "val_data.feather"))
    elif category == 'test':
        data = pd.read_feather(os.path.join(base_dirs, "test_data.feather"))

    data.rename(columns={ret_name: "nxt1_ret"}, inplace=True)
    data = data[['trade_time', 'code', 'nxt1_ret'] + features + regime]
    data = data.sort_values('trade_time').reset_index(drop=True)
    data = _sanitize_frame(data, ['nxt1_ret'] + features + regime)
    if data['code'].nunique() != 1:
        raise ValueError(
            f"test_data 不是单标的，检测到 {data['code'].nunique()} 个 code")
    return data


instruments = 'rbb'
method = 'ricso2'
period = 5
task_id = '113001'
model_id = '1018806311332385'
category = 'test'
pdb.set_trace()

mongo_client = MongoDBManager(
    uri='mongodb://neutron:6oZZ5emy@10.200.122.41:37240/neutron')
repo = Repository(mongo_client=mongo_client)
pdb.set_trace()

state_key = make_state_key(
    task_id=task_id,
    code=INSTRUMENTS_CODES[instruments],
    symbol="{0}9999".format(instruments.lower()),
    period=period,
    workflow_name="sirius0003",
    rule_name="TradeOnline2",
    rule_version="v1",
)

doc = repo.load(state_key)
pdb.set_trace()

trader = repo.load_state(
    state_key=state_key,
    state_cls=TradeOnline2,
    default_hold_bars=int(period),
)

output_dir = os.path.join(base_path, method, instruments, 'temp', 'model',
                          str(task_id), str(period), 'rl', 'composite',
                          "model", "rl", str(model_id), "data")

factors_infos, params = load_sirius_params(code=INSTRUMENTS_CODES[instruments],
                                           task_id=str(model_id))

pdb.set_trace()
workflow = WorkFlow(directory=params['model_path'],
                    code=INSTRUMENTS_CODES[instruments],
                    symbol="{0}9999".format(instruments.lower()),
                    task_id=task_id,
                    factors_infos=factors_infos,
                    softmax_temperature=params['softmax_temperature'],
                    min_open_signal_abs=params['min_open_signal_abs'],
                    method=params['method'],
                    win=params['win'],
                    period=params['horizon'],
                    signal_method=params['signal_method'],
                    signal_params=params['signal_params'],
                    trader=trader)

data = load_data0(
    method=method,
    instruments=instruments,
    period=period,
    task_id=task_id,
    ret_name="nxt1_ret_{0}h".format(1),
    features=workflow.features,
    regime=[],
    category=category)

total_data1 = data.set_index(['trade_time', 'code'])
all_trade_times = total_data1.index.get_level_values(
    'trade_time').unique().sort_values()
pdb.set_trace()
res = []
for time in all_trade_times[0:2000]:
    print(time)
    rt = workflow.create_values(trade_time=time, data=total_data1)
    events = workflow.conversion_signals(trade_time=time,
                                         raw_action=pd.DataFrame([rt]),
                                         name='net_er_out')
    res.extend(events)

    repo.save_state(state_key=state_key, state=workflow.trader)

pdb.set_trace()
print('-->')
