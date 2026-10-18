### 指定表达式，批量生成因子绩效
import itertools
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from lumina.genetic.util import create_id
from lib.aux001 import calc_expression
from lib.cux001 import FactorEvaluate1, generate_simple_id
from lumina.genetic.process import *
from lib.iux001 import fetch_data, merging_data1
from kdutils.tactix import Tactix
from kdutils.macro2 import *


def programs_metrics(column, total_data, total_data1, period, outputs):
    print(column)
    factor_data = calc_expression(expression=column, total_data=total_data1)
    dt = merging_data1(factor_data=factor_data,
                       returns_data=total_data,
                       period=period)
    evaluate1 = FactorEvaluate1(factor_data=dt,
                                factor_name='transformed',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=15,
                                fee=0.000,
                                scale_method='roll_zscore',
                                resampling_win=period,
                                expression=column)
    state_dt = evaluate1.run()
    data = evaluate1.resample_data.reset_index()
    data.name = column
    name_id = create_id(generate_simple_id(column))
    filename = os.path.join(outputs, "{}.feather".format(name_id))
    state_dt['id'] = name_id
    state_dt['name'] = column
    data.to_feather(filename)
    return state_dt


@add_process_env_sig
def run_metrics(target_column, total_data, total_data1, period, outputs):
    return run_process(target_column=target_column,
                       callback=programs_metrics,
                       total_data=total_data,
                       total_data1=total_data1,
                       period=period,
                       outputs=outputs)


###计算绩效
def evalute_metrics(method,
                    instruments,
                    period,
                    task_id,
                    session,
                    index,
                    expressions,
                    datasets=['train', 'val']):

    total_data = fetch_data(method=method,
                            task_id=task_id,
                            instruments=instruments,
                            datasets=datasets)
    total_data1 = total_data.set_index(['trade_time'])
    pdb.set_trace()
    dirs = os.path.join(base_path, method, instruments, 'metrics',
                        str(task_id), "nxt1_ret_{}h".format(str(period)),
                        str(session), str(index))

    outputs = os.path.join(dirs, "sequence")
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    ## 多进程计算绩效, 表达式转化为 id 存储
    k_split = 4
    expression_list = expressions
    process_list = split_k(k_split, expression_list)
    res = create_parellel(process_list=process_list,
                          callback=run_metrics,
                          period=period,
                          total_data=total_data,
                          total_data1=total_data1,
                          outputs=outputs)
    res1 = list(itertools.chain.from_iterable(res))
    results = pd.DataFrame(res1)
    results.to_csv(os.path.join(dirs, "metrics.csv"))


def metrics1(method,
             instruments,
             period,
             task_id,
             session,
             index,
             datasets=['train', 'val']):
    expressions = [
    "MRANK(30, SUBBED(MRANK(25, DELTA(5, 'close')), MRANK(25, DELTA(5, 'volume'))))",
    "MRANK(25, MCORR(35, DELTA(5, 'close'), SHIFT(5, DELTA(5, 'close'))))",
    "MRANK(30, ADDED(DIV(SUBBED(EMA(7, 'close'), EMA(15, 'close')), MSTD(25, 'close')), DIV(SUBBED(EMA(15, 'close'), EMA(40, 'close')), MSTD(40, 'close'))))",
    "MRANK(25, EMA(12, ADDED(ADDED('depth_imbalance_0', 'depth_imbalance_1'), ADDED('depth_imbalance_2', 'depth_imbalance_3'))))",
    "MRANK(30, MCORR(30, 'money', DIV('money', 'volume')))",
    "MRANK(35, SUBBED(DIV('money', 'volume'), EMA(30, DIV('money', 'volume'))))",
    "MRANK(40, DIV(SUBBED('high', 'low'), EMA(35, SUBBED('high', 'low'))))",
    "MRANK(30, DELTA(7, 'order_flow_imbanlace_weighted5'))",
    "MRANK(30, MKURT(30, 'pct_change'))",
    "MRANK(30, ADDED(MUL(0.5, 'order_flow_imbanlace_1'), MUL(0.3, 'order_flow_imbanlace_avg5')))",
    "MRANK(35, MUL(DELTA(10, 'close'), DELTA(10, 'volume')))",
    "MRANK(35, EMA(18, 'mci_imbalance'))",
    "ADDED(ADDED(MRANK(35, DIV(DELTA(7, 'close'), MSTD(35, 'close'))), MRANK(35, EMA(10, 'net_money_in'))), MRANK(35, EMA(7, 'depth_imbalance_0')))",
    "SUBBED(MRANK(35, EMA(12, 'net_money_in')), MRANK(35, 'pct_change'))",
    "MRANK(45, DIV(EMA(12, 'volume'), EMA(40, 'volume')))",
    "MRANK(30, DELTA(10, EMA(12, 'net_money_in')))",
    "MRANK(30, DELTA(12, RSI(14, 'close')))",
    "MRANK(30, ADDED(MSKEW(35, 'pct_change'), MSKEW(18, 'pct_change')))",
    "MRANK(30, MSKEW(35, MSTD(7, 'pct_change')))",
    "MRANK(35, DIV(SUBBED('realized_volatility', EMA(35, 'realized_volatility')), MSTD(35, 'realized_volatility')))",
    "MRANK(30, DIV(SUBBED('money', EMA(30, 'money')), MSTD(30, 'money')))",
    "MRANK(30, EMA(12, 'order_imbalance_ratio5'))",
    "MRANK(40, DIV(EMA(18, 'pct_change'), MSTD(45, 'pct_change')))",
    "MRANK(35, DIV(MSTD(18, 'volume'), EMA(40, 'volume')))",
    "MRANK(35, DIV(DELTA(10, 'twap'), MSTD(40, 'twap')))",
    "MRANK(35, DIV(SUBBED(EMA(10, 'close'), EMA(35, 'close')), MSTD(35, 'close')))",
    "MRANK(30, EMA(18, 'ask_bid_press'))",
    "MRANK(30, DELTA(12, MQUANTILE(45, 'close')))",
    "MRANK(30, MCORR(30, EMA(12, 'close'), EMA(35, 'close')))",
    "MRANK(30, MRes(45, EMA(18, 'close'), 'close'))",
    "MRANK(30, DIV(SUBBED('smart_volume_in', 'smart_volume_out'), MSTD(18, 'volume')))",
    "MRANK(25, EMA(10, ADDED(MUL(0.4, 'depth_imbalance_0'), ADDED(MUL(0.3, 'depth_imbalance_1'), MUL(0.2, 'depth_imbalance_2')))))",
    "MRANK(40, ADDED(DIV(DELTA(10, 'close'), MSTD(30, 'close')), DIV(DELTA(25, 'close'), MSTD(55, 'close'))))",
    "MRANK(30, MADecay(35, 'pct_change'))",
    "MRANK(25, EMA(10, ADDED('price_imbalance_0', 'price_imbalance_1')))",
    "MRANK(30, MCoef(40, EMA(18, 'close'), 'close'))",
    "MRANK(30, MCORR(35, EMA(5, 'net_money_in'), DELTA(10, 'close')))",
    "MRANK(30, EMA(15, 'mid_price_bias_ratio'))",
    "ADDED(ADDED(MRANK(35, MCORR(35, DELTA(5, 'close'), SHIFT(5, DELTA(5, 'close')))), MRANK(35, EMA(12, 'net_money_in'))), MRANK(35, EMA(10, 'depth_imbalance_0')))",
    "ADDED(ADDED(MRANK(35, SUBBED(MRANK(25, DELTA(5, 'close')), MRANK(25, DELTA(5, 'volume')))), MRANK(35, MCORR(30, 'money', DIV('money', 'volume')))), MRANK(35, EMA(12, ADDED(ADDED('depth_imbalance_0', 'depth_imbalance_1'), ADDED('depth_imbalance_2', 'depth_imbalance_3')))))"
]
    evalute_metrics(method=method,
                    instruments=instruments,
                    period=period,
                    task_id=task_id,
                    session=session,
                    index=index,
                    expressions=expressions,
                    datasets=datasets)


if __name__ == '__main__':
    variant = Tactix().start()
    metrics1(method=variant.method,
             instruments=variant.instruments,
             period=variant.period,
             task_id=variant.task_id,
             session=variant.session,
             index=variant.index)
