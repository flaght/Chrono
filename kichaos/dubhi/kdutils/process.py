import pdb
import numpy as np
import pandas as pd
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


def factor_score_sig(invar):
    invar_rank = invar.rank(axis=1, method='max')
    count = invar_rank.count(axis=1)
    invar_rank = (invar_rank - 3. / 8.).div(count + 1. / 4., axis='rows')
    invar_score = pd.DataFrame(norm.ppf(invar_rank),
                               index=invar.index,
                               columns=invar.columns)
    invar_score_top = invar_score.copy()
    invar_score_top[invar_score_top < 0] = np.nan
    invar_score_bot = invar_score.copy()
    invar_score_bot[invar_score_bot >= 0] = np.nan
    return invar_score, invar_score_top, invar_score_bot


def CalRet(dummy_tradable,
           weightM,
           ret,
           ret_c2o,
           ret_benchmark,
           freq,
           skip=0,
           cost=0.003,
           leveage=None):
    out = pd.Series(
        index=['ret', 'std', 'sharpe', 'turnover', 'maxdd', 'calmar'],
        dtype='float')
    weightM[weightM == 0] = np.nan
    weightM = weightM.shift(skip)

    weightMbefore = weightM.shift(1)
    weightM[dummy_tradable.isna()] = weightMbefore[dummy_tradable.isna()]

    tvs = abs(weightM.sub(weightM.shift(1), fill_value=0)).sum(
        axis=1, min_count=0) * 0.5
    if ret_c2o is not None:
        tvs = tvs.shift(1)

    weightM = weightM.shift(1)
    dweight = weightM.sub(weightM.shift(1), fill_value=0)

    if ret_c2o is None:
        ret_c2o = 0
    if ret_benchmark is not None:
        usebenchmark = ret_benchmark.copy()
        usebenchmark[weightM.count(axis=1) == 0] = 0
    else:
        usebenchmark = 0

    if leveage is None:
        leveage = 1
    pnl = np.log(((np.exp(ret) - 1) * weightM).sum(axis=1, min_count=0) -
                 ((np.exp(ret_c2o) - 1) * dweight).sum(axis=1, min_count=0) -
                 (np.exp(usebenchmark) - 1) - tvs * cost + 1) * leveage

    tv = tvs.mean()
    retL = pnl.mean() * freq
    retL_std = pnl.std() * np.sqrt(freq)
    sharpL = retL / retL_std
    pnlL = pnl.cumsum()
    maxdd = pnlL.expanding().max() - pnlL
    maxddL = maxdd.max()
    ret2mddL = retL / maxddL
    out['ret'] = retL
    out['std'] = retL_std
    out['sharpe'] = sharpL
    out['turnover'] = tv
    out['maxdd'] = maxddL
    out['calmar'] = ret2mddL
    return out, pnl, tvs


# mode 0:等权
def TopNWeight(vardummy, indata, hold, n, mode=0, trans=False):
    pdb.set_trace()
    invar = indata * vardummy
    returndata = invar.apply(lambda x: x[x.nlargest(n, keep='all').index],
                             axis=1)
    returndata[~returndata.isna()] = 1
    returndata = returndata.reindex(index=vardummy.index,
                                    columns=vardummy.columns)
    if mode != 0:
        returndata = returndata * indata
        if trans:
            returndata, _, _ = factor_score_sig(returndata)
            returndata = returndata.sub(returndata.min(axis=1),
                                        axis='rows') + 0.0001
    returndata = returndata.div(returndata.sum(min_count=1, axis=1),
                                axis='rows')
    returndata = returndata.rolling(window=hold, min_periods=1).sum() / hold
    return returndata
