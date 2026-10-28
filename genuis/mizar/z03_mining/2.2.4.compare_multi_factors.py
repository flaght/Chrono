import os, itertools, pdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix
from kdutils.macro2 import base_path, INSTRUMENTS_CODES
from ultron.factor.genetic.geneticist.operators import *
from lumina.genetic.process import *
from lib.iux001 import fetch_data
from lib.iux002 import calc_all1, generate_simple_id, create_id
from lib.iux003 import normalize_assets, comparison_rows, MultiAssetFactorComparator


def create_evalute(column, period, asset_data, asset_specs, outputs):
    evaluations = {}
    for asset in asset_specs:
        symbol = asset['instruments']
        try:
            # 与双品种版本一样，直接复用 calc_all1，不另写评估算法。
            evaluations[symbol] = calc_all1(expression=column,
                                            total_data1=asset_data[symbol],
                                            period=period)
        except Exception as exc:
            print('===> {}: {}'.format(symbol, exc))
    rows = comparison_rows(evaluations=evaluations,
                           asset_specs=asset_specs,
                           asset_data=asset_data,
                           expression=column,
                           period=period)
    fc = MultiAssetFactorComparator(evaluations=evaluations,
                                    rows=rows,
                                    expression=column,
                                    period=period)
    fc.plot_comparison()
    fc.save_results(base_output_dir=outputs)


@add_process_env_sig
def run_evalute(target_column, period, asset_data, asset_specs, outputs):
    return run_process(target_column=target_column,
                       callback=create_evalute,
                       period=period,
                       asset_data=asset_data,
                       asset_specs=asset_specs,
                       outputs=outputs)


def fetch_data1(method, instruments, datasets, features, task_id, period):
    total_data = fetch_data(method=method,
                            instruments=instruments,
                            task_id=task_id,
                            datasets=datasets)
    mix_columns = list(set(features) & set(total_data.columns))
    if len(mix_columns) <= 0:
        print("not any features!!!!")
        return pd.DataFrame()

    total_data = total_data[['trade_time', 'code'] + mix_columns +
                            ['nxt1_ret_{}h'.format(period)]]
    return total_data


def load_asset_data(method, asset_specs, datasets, features, period):
    asset_data, data_errors = {}, {}
    for asset in asset_specs:
        symbol = asset['instruments']
        try:
            asset_data[symbol] = fetch_data1(method=method,
                                             instruments=symbol,
                                             datasets=datasets,
                                             features=features,
                                             task_id=asset['task_id'],
                                             period=period)
        except Exception as exc:
            # 没有文件或字段错误与空历史分别记录，其他品种继续比较。
            asset_data[symbol] = pd.DataFrame(columns=['trade_time'])
            data_errors[symbol] = '{}: {}'.format(type(exc).__name__, exc)
            print('===> {}: {}'.format(symbol, exc))
    return asset_data, data_errors


def load_factors(method,
                 instruments,
                 period,
                 task_id,
                 session,
                 category='gentic'):
    dirs = os.path.join(base_path, method, instruments, category, 'ic',
                        str(task_id), 'nxt1_ret_{}h'.format(period),
                        str(session))
    filename = os.path.join(dirs,
                            'programs_{}_{}.feather'.format(task_id, session))
    if not os.path.exists(filename):
        print('No factors file the criteria: {}'.format(filename))
        return pd.DataFrame()
    programs = pd.read_feather(filename)
    if 'formual' not in programs and 'formula' in programs:
        programs = programs.rename(columns={'formula': 'formual'})
    return programs[programs['final_fitness'] > 0.001][[
        'name', 'formual', 'final_fitness'
    ]].drop_duplicates('formual').copy()


def run2(method,
         instruments,
         period,
         task_id,
         session,
         asset_specs,
         datasets=['train', 'val']):
    pdb.set_trace()
    asset_specs = normalize_assets(asset_specs, INSTRUMENTS_CODES)
    left_symbol = instruments
    outputs = os.path.join("records", method, left_symbol, 'rulex',
                           str(task_id), "nxt1_ret_{}h".format(str(period)),
                           str(session))
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    programs = load_factors(method=method,
                            instruments=instruments,
                            period=period,
                            task_id=task_id,
                            session=session,
                            category='eligible')

    if programs.empty:
        print('No factors data the criteria')
        return outputs

    unique_exprs = programs['formual'].tolist()
    expr_to_id = {
        expr: create_id(generate_simple_id(expr))
        for expr in unique_exprs
    }
    programs['id'] = programs['formual'].map(expr_to_id)
    res1 = []
    for row in programs.itertuples():
        filename = os.path.join(outputs, row.id, 'performance_summary.csv')
        plot = os.path.join(outputs, row.id, 'comparison_plot.png')
        if os.path.exists(filename) and os.path.exists(plot):
            if pd.read_csv(filename)['status'].eq('ok').all():
                res1.append(row.id)
    programs = programs[~programs['id'].astype(str).isin(res1)].reset_index(
        drop=True)
    if programs.empty:
        return outputs

    features = [
        eval(program.formual)._dependency for program in programs.itertuples()
    ]
    features = list(itertools.chain.from_iterable(features))
    features = list(set(features))
    
    asset_data, data_errors = load_asset_data(method=method,
                                              asset_specs=asset_specs,
                                              datasets=datasets,
                                              features=features,
                                              period=period)

    process_list = split_k(2, programs['formual'].tolist())
    results = create_parellel(process_list=process_list,
                              callback=run_evalute,
                              period=period,
                              asset_data=asset_data,
                              asset_specs=asset_specs,
                              outputs=outputs)
    


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'all':
        run2(method=variant.method,
             instruments=variant.instruments,
             period=variant.period,
             task_id=variant.task_id,
             asset_specs=variant.asset_specs,
             session=variant.session)
