import pandas as pd
import os, argparse
from dotenv import load_dotenv

load_dotenv()

from lumina.genetic.process import *
from kdutils.common import fetch_temp_data, fetch_temp_returns
from lib.cux002 import *
from lib.cux003 import *

leg_mappping = {"ims": ["ics"]}

expressions = [
    "EMA(2,MCPS(2,'oi034_1_2_1'))", "EMA(2,MCPS(2,EMA(2,'oi034_1_2_1')))",
    "MCPS(2,'oi034_1_2_1')",
    "MRes(3,MMinDiff(2,'oi022_1_2_1'),MPWMA(4,'tv004_1_2_1','rv001_2_3_0_2'))",
    "MRes(3,'tc015_1_2_1','tn003_1_1_2_3_1')",
    "MRes(3,'tc015_1_2_1',EMA(3,'tn003_1_1_2_3_1'))", "MDPO(3,'tv017_1_2_1')",
    "MCPS(2,'oi034_1_2_1')", "EMA(2,MCPS(2,EMA(2,'oi034_1_2_1')))",
    "MDEMA(2,'tn003_2_1_2_3_1')", "MDPO(4,'tc005_1_1_2_1')",
    "MRes(4,SIGN(SIGLOG10ABS(MMAX(3,MRes(4,'tn008_1_2_1_1',SIGLOG10ABS('tc001_2_3_0'))))),'tc014_1_1_2_0')",
    "MRes(4,SIGN(SIGLOG10ABS('tc001_2_3_0')),'tc014_1_1_2_0')",
    "MDIFF(2,'tn003_1_1_2_3_1')", "SIGMOID(MADiff(3,'iv012_1_2_1'))",
    "MMeanRes(4,'dv005_1_2_1','ixy008_1_2_1')",
    "MDIFF(2,ADDED('ixy007_1_2_0','tv017_1_2_1'))",
    "MDEMA(2,MDEMA(2,'tc008_1_2_0'))",
    "MADiff(4,SUBBED('tn008_1_2_1_4',MRes(4,'tn009_1_2_0_4','ixy007_1_2_1')))",
    "EMA(3,ABS(MPERCENT(3,'tc006_1_2_0')))", "MT3(2,'ixy006_1_2_1')",
    "MINIMUM(MDEMA(2,MMIN(2,ABS(MSUM(4,'tn008_1_2_0_4')))),'ixy007_1_2_1')",
    "SIGMOID(MDPO(3,MADiff(3,SIGMOID(SIGMOID('iv012_1_2_1')))))",
    "WMA(4,'ixy006_1_2_1')", "MADiff(2,'ixy007_1_2_1')",
    "EMA(3,MCPS(4,'oi034_1_2_1'))", "MCPS(3,SIGLOG2ABS('tn003_2_1_2_3_1'))",
    "MDPO(4,MADecay(3,EMA(2,'tc005_1_1_2_1')))", "SIGMOID('tc013_1_2_1')"
]


def create_evalute(column, period, left_data, right_data, left_symbol,
                   right_symbol, outputs):
    left_evaluate = calc_all(expression=column,
                             total_data1=left_data,
                             period=period)
    right_evaluate = calc_all(expression=column,
                              total_data1=right_data,
                              period=period)
    fc = FactorComparator(eval_left=left_evaluate,
                          eval_right=right_evaluate,
                          left_name=left_symbol,
                          right_name=right_symbol,
                          expression=column)
    fc.plot_comparison()
    fc.save_results(base_output_dir=outputs)


@add_process_env_sig
def run_evalute(target_column, period, left_data, right_data, left_symbol,
                right_symbol, outputs):
    status_data = run_process(target_column=target_column,
                              callback=create_evalute,
                              period=period,
                              left_data=left_data,
                              right_data=right_data,
                              left_symbol=left_symbol,
                              right_symbol=right_symbol,
                              outputs=outputs)
    return status_data


def run1(method, instruments, period, task_id):
    left_symbol = instruments
    right_symbol = leg_mappping[instruments][0]
    left_data = fetch_data(method=method,
                           instruments=left_symbol).set_index('trade_time')
    right_data = fetch_data(method=method,
                            instruments=right_symbol).set_index('trade_time')

    outputs = os.path.join("records", "select", method, task_id, left_symbol)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    k_split = 4
    expression_list = expressions[4:]
    pdb.set_trace()
    process_list = split_k(k_split, expression_list)
    res = create_parellel(process_list=process_list,
                          callback=run_evalute,
                          period=period,
                          left_data=left_data,
                          right_data=right_data,
                          left_symbol=left_symbol,
                          right_symbol=right_symbol,
                          outputs=outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--dethod', type=str, default='ic', help='data method')

    parser.add_argument('--method',
                        type=str,
                        default='bicso0',
                        help='data method')
    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instrument')

    parser.add_argument('--task_id',
                        type=str,
                        default='200036',
                        help='code or instruments')

    parser.add_argument('--period',
                        type=str,
                        default='15',
                        help='code or instruments')

    args = parser.parse_args()

    run1(method=args.method,
         instruments=args.instruments,
         period=args.period,
         task_id=args.task_id)
