## 相关性计算  1.收益率/因子值  2.排序指标(多个，按顺序) 3.阈值
import itertools
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from ultron.ump.similar.corrcoef import ECoreCorrType
from lumina.genetic.util import create_id
from lumina.genetic.process import *
from ultron.ump.similar.corrcoef import corr_xy
from lib.iux001 import fetch_data, merging_data1
from lib.aux001 import calc_expression
from lib.cux001 import FactorEvaluate1, generate_simple_id
from kdutils.tactix import Tactix
from kdutils.macro2 import *

leg_mappping = {"rbb": ["hcb"], "ims": ["ics"]}

sort_mapping = {
    "1": ["avg_ret", "abs_ic", "calmar"],
}


## 加载选中
def fetch_chosen_factors(method, instruments, task_id, period):
    filename = os.path.join(base_path, method, instruments, "rulex",
                            str(task_id), "nxt1_ret_{0}h".format(period),
                            "draft.csv")
    expressions = pd.read_csv(filename).to_dict(orient='records')
    expressions = {item['formula']: item for item in expressions}
    expressions = list(expressions.values())
    return expressions


def programs_metrics(column, total_data, total_data1, period, outputs):
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
    data = evaluate1.factor_data.reset_index()
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
                    lnstruments,
                    period,
                    task_id,
                    programs,
                    category=None,
                    sort_index=None,
                    threshold=None,
                    datasets=['train', 'val']):

    total_data = fetch_data(method=method,
                            task_id=task_id,
                            instruments=lnstruments,
                            datasets=datasets)
    total_data1 = total_data.set_index(['trade_time'])
    pdb.set_trace()
    dirs = os.path.join(base_path, method, instruments, 'correlation',
                        str(task_id), "nxt1_ret_{}h".format(str(period)),
                        lnstruments)

    outputs = os.path.join(dirs, "sequence")
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    ## 多进程计算绩效, 表达式转化为 id 存储
    k_split = 4
    expression_list = [program['formula'] for program in programs]
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
             category=None,
             sort_index=None,
             threshold=None,
             datasets=['train', 'val']):
    programs = fetch_chosen_factors(method=method,
                                    instruments=instruments,
                                    task_id=task_id,
                                    period=period)
    evalute_metrics(method=method,
                    instruments=instruments,
                    lnstruments=instruments,
                    period=period,
                    task_id=task_id,
                    datasets=datasets,
                    programs=programs)


def metrics2(method,
             instruments,
             period,
             task_id,
             category=None,
             sort_index=None,
             threshold=None,
             datasets=['train', 'val']):
    programs = fetch_chosen_factors(method=method,
                                    instruments=instruments,
                                    task_id=task_id,
                                    period=period)
    evalute_metrics(method=method,
                    instruments=instruments,
                    lnstruments=leg_mappping[instruments][0],
                    period=period,
                    task_id=task_id,
                    datasets=datasets,
                    programs=programs)


def load_sequence(dirs, category, ids):
    res = []
    for id in ids:
        filename = os.path.join(dirs, "sequence", "{0}.feather".format(id))
        sequence_data = pd.read_feather(filename)
        sequence_data = sequence_data.set_index('trade_time')[category]
        sequence_data.name = id
        res.append(sequence_data)
    total_data = pd.concat(res, axis=1)
    columns = total_data.columns
    filter_col = []
    for col in columns:
        cover_rate = 1 - (total_data[total_data[col].isna()].shape[0] /
                          total_data[col].shape[0])
        if cover_rate < 0.8:
            filter_col.append(col)
    total_data = total_data.drop(filter_col, axis=1)
    return total_data.fillna(method="ffill")


def correlation(method,
                instruments,
                lnstruments,
                period,
                task_id,
                category=None,
                sort_index=None,
                threshold=None,
                datasets=['train', 'val']):
    dirs = os.path.join(base_path, method, instruments, 'correlation',
                        str(task_id), "nxt1_ret_{}h".format(str(period)),
                        lnstruments)
    metrics_data = pd.read_csv(os.path.join(dirs, "metrics.csv"), index_col=0)
    metrics_data['abs_ic'] = np.abs(metrics_data['ic_mean'])
    metrics_data = metrics_data.sort_values(by=sort_mapping["1"],
                                            ascending=False)
    ## 加载收益率
    returns_data = load_sequence(dirs=dirs,
                                 category=category,
                                 ids=metrics_data['id'])
    sort_id = [id for id in metrics_data['id'] if id in returns_data.columns]
    returns_data = returns_data[sort_id]
    filter_cols = []
    for i in range(0, len(sort_id) - 1):
        for j in range(i + 1, len(sort_id)):
            x_col = sort_id[i]
            y_col = sort_id[j]
            if x_col in filter_cols or y_col in filter_cols or x_col == y_col:
                continue
            #returns_data[x_col].corr(returns_data[y_col], method='spearman')
            corr = corr_xy(returns_data[x_col], returns_data[y_col],
                           ECoreCorrType.E_CORE_TYPE_SPERM)
            print(x_col, y_col, corr)
            if corr > threshold:
                filter_cols.append(y_col)
    returns_data = returns_data.drop(filter_cols, axis=1)
    keep_columns = returns_data.columns
    keep_metrics = metrics_data[metrics_data.id.isin(keep_columns)]

    programs = fetch_chosen_factors(method=method,
                                    instruments=instruments,
                                    task_id=task_id,
                                    period=period)
    programs = pd.DataFrame(programs).rename(columns={'formula': 'name'})
    keep_programs = keep_metrics.merge(programs, on=['name'])
    keep_programs['direction1'] = np.where(keep_programs['ic_mean'] > 0, 1, -1)
    check_count = keep_programs[keep_programs['direction1'] ==
                                keep_programs['direction']].shape[0]
    pdb.set_trace()
    if check_count != keep_programs.shape[0]:
        pdb.set_trace()
        print("check count ")

    filename = os.path.join(
        base_path, method, instruments, "rulex", str(task_id),
        "nxt1_ret_{0}h".format(period),
        "chosen_{0}_{1}_{2}.csv".format(category, str(sort_index),
                                        str(int(threshold * 100))))
    pdb.set_trace()
    keep_programs[['id', 'name', 'direction']].rename(columns={
        'name': 'formula'
    }).to_csv(filename, encoding="UTF-8")


def correlation1(method,
                 instruments,
                 period,
                 task_id,
                 category=None,
                 sort_index=None,
                 threshold=None,
                 datasets=['train', 'val']):
    correlation(method=method,
                instruments=instruments,
                lnstruments=instruments,
                period=period,
                task_id=task_id,
                category=category,
                sort_index=sort_index,
                threshold=threshold,
                datasets=datasets)


def correlation2(method,
                 instruments,
                 period,
                 task_id,
                 category=None,
                 sort_index=None,
                 threshold=None,
                 datasets=['train', 'val']):
    correlation(method=method,
                instruments=instruments,
                lnstruments=leg_mappping[instruments][0],
                period=period,
                task_id=task_id,
                category=category,
                sort_index=sort_index,
                threshold=threshold,
                datasets=datasets)


def correlation1(method,
                 instruments,
                 period,
                 task_id,
                 category=None,
                 sort_index=None,
                 threshold=None,
                 datasets=['train', 'val']):
    correlation(method=method,
                instruments=instruments,
                lnstruments=instruments,
                period=period,
                task_id=task_id,
                category=category,
                sort_index=sort_index,
                threshold=threshold,
                datasets=datasets)


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == "metrics1":
        metrics1(method=variant.method,
                 instruments=variant.instruments,
                 period=variant.period,
                 task_id=variant.task_id,
                 category=variant.category,
                 sort_index=variant.sort_index,
                 threshold=variant.threshold)
    elif variant.form == "metrics2":
        metrics2(method=variant.method,
                 instruments=variant.instruments,
                 period=variant.period,
                 task_id=variant.task_id,
                 category=variant.category,
                 sort_index=variant.sort_index,
                 threshold=variant.threshold)
    elif variant.form == "correlation1":
        correlation1(method=variant.method,
                     instruments=variant.instruments,
                     period=variant.period,
                     task_id=variant.task_id,
                     category=variant.category,
                     sort_index=variant.sort_index,
                     threshold=variant.threshold)
    elif variant.form == "correlation2":
        correlation2(method=variant.method,
                     instruments=variant.instruments,
                     period=variant.period,
                     task_id=variant.task_id,
                     category=variant.category,
                     sort_index=variant.sort_index,
                     threshold=variant.threshold)
