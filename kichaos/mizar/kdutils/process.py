import pdb
from ultron.kdutils.progress import Progress
from ultron.ump.core.process import add_process_env_sig, EnvProcess
from ultron.kdutils.parallel import delayed, Parallel


def split_k(k_split, columns):
    if len(columns) < k_split:
        return [[col] for col in columns]
    sub_column_cnt = int(len(columns) / k_split)
    group_adjacent = lambda a, k: zip(*([iter(a)] * k))
    cols = list(group_adjacent(columns, sub_column_cnt))
    residue_ind = -(len(columns) % sub_column_cnt) if sub_column_cnt > 0 else 0
    if residue_ind < 0:
        cols.append(columns[residue_ind:])
    return cols


def run_process(target_column,
                callback,
                label='predict groups model',
                **kwargs):
    res = []
    with Progress(len(target_column), 0, label=label) as pg:
        for i, column in enumerate(target_column):
            data = callback(column=column, **kwargs)
            res.append(data)
            pg.show(i + 1)
    return res


def create_parellel(process_list, callback, **kwargs):
    parallel = Parallel(len(process_list), verbose=0, pre_dispatch='2*n_jobs')

    res = parallel(
        delayed(callback)(
            target_column=target_column, env=EnvProcess(), **kwargs)
        for target_column in process_list)
    return res
