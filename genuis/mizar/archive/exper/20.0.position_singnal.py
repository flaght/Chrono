import os
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from lib.bck002 import rebuild_executed_signal_for_eval1
from lib.cux001 import FactorEvaluate1

method = 'ricso2'
instruments = 'rbb'
task_id = '113001'
period = 5


def rebuild_executed_signal_for_eval1(signal_data: pd.DataFrame,
                                     position_data: pd.DataFrame,
                                     signal_col: str = "signal") -> pd.DataFrame:
    """
    从 build_capped_locked_signals 生成的 position_data 中，
    只提取真正由模型触发的 open_exposure，
    再映射回原始 signal_data 的完整分钟时间轴。

    注意：
    - open_exposure 才是模型信号触发的开暴露事件
    - restore_lock 是到期恢复对锁，不是预测信号，不能放入 FactorEvaluate1
    - 输出必须保留完整 trade_time/code 网格，未开仓的位置 signal=0
    """
    base = signal_data.copy()
    base["trade_time"] = pd.to_datetime(base["trade_time"])

    if signal_col not in base.columns:
        raise ValueError(f"signal_data missing column: {signal_col}")

    # 保留原始信号，便于后续对比
    base["_raw_signal"] = base[signal_col].fillna(0).astype(int)

    opens = position_data.copy()
    opens["trade_time"] = pd.to_datetime(opens["trade_time"])

    opens = opens[opens["signal_type"] == "open"].copy()

    if opens.empty:
        out = base.copy()
        out[signal_col] = 0
        return out

    # 如果同一时刻有多条 open_exposure，合并成一个方向。
    # 正常单标的一分钟最多一条，但这里做防御处理。
    opens = (
        opens.groupby(["trade_time", "code"], as_index=False)
        .agg(
            executed_direction=("direction", "sum"),
            executed_lots=("numbers", "sum"),
        )
    )

    opens["executed_signal"] = np.sign(opens["executed_direction"]).astype(int)

    out = base.merge(
        opens[["trade_time", "code", "executed_signal", "executed_lots"]],
        on=["trade_time", "code"],
        how="left",
    )

    out[signal_col] = out["executed_signal"].fillna(0).astype(int)
    out["executed_lots"] = out["executed_lots"].fillna(0).astype(int)

    return out


basic_path = os.path.join(base_path, method, instruments, 'temp', 'model',
                          str(task_id), str(period), 'rl', 'backtest', 'rl',
                          '1013836755991964')
filename = os.path.join(basic_path, '1018806311332385', 'erband_signal',
                        '1002_test', 'position_data.feather')
postions_data = pd.read_feather(filename)

base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                          str(task_id), str(period), 'rl')
dirs1 = os.path.join(base_path1, "signal", "rl", str('1018806311332385'))
filename = os.path.join(dirs1, 'erband_signal', '1002_test.feather')
signal_data = pd.read_feather(filename)

executed_signal = rebuild_executed_signal_for_eval1(
    signal_data=signal_data,
    position_data=postions_data,  # 原始完整 position_data，不要提前过滤列
    signal_col="signal",
)
pdb.set_trace()
executed_signal = executed_signal[[
    'trade_time', 'code', 'signal', 'nxt1_ret_5h'
]]

# evaluate1 = FactorEvaluate1(factor_data=signal_data,
#                                    factor_name='signal',
#                                    ret_name='nxt1_ret_{0}h'.format(period),
#                                    roll_win=15,
#                                    fee=0.0,
#                                    scale_method='raw',
#                                    expression="signal",
#                                    resampling_win=period,
#                                    name="signal")
# _ = evaluate1.run()
# evaluate1.plot_results()

pdb.set_trace()
evaluate2 = FactorEvaluate1(factor_data=executed_signal,
                            factor_name='signal',
                            ret_name='nxt1_ret_{0}h'.format(period),
                            roll_win=15,
                            fee=0.0,
                            scale_method='raw',
                            expression="postions",
                            resampling_win=period,
                            name="postions")
_ = evaluate2.run()
evaluate2.plot_results()
