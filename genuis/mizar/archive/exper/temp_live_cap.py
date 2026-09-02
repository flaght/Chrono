import json,pdb,os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


from lib.bck002 import live_capped_locked_step


STATE_PATH = "live_locked_state.json"


def load_state(path=STATE_PATH):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state, path=STATE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


pdb.set_trace()
params = {
    "base_position": 10,
    "lot_per_signal": 1,
    "hold_bars": 5,
    "entry_resampling_win": 5,
    "date_col": "trade_date",
    "max_daily_open_lots": 12,
    "max_daily_open_lots_per_direction": 6,
    "max_active_open_lots": 1,
    "max_active_open_lots_per_direction": 1,
    "min_abs_value": None,
    "block_same_direction_reentry": True,
    "block_opposite_direction_reentry": True,
    "extend_same_direction": True,
}

bar_signal = {
    "trade_time": "2026-06-12 09:35:00",
    "trade_date": "2026-06-12",
    "code": "RB",
    "signal": 1,
    "value": 0.18,
}

state = load_state()

events, state = live_capped_locked_step(
    bar_signal=bar_signal,
    state=state,
    **params,
)

save_state(state)