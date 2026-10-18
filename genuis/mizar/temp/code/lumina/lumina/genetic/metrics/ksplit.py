from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import empyrical
import pdb

def time_series(returns_series, n_splits, max_train_size=None):
    all_folds_performance = []
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)
    for _, (_, val_idx) in enumerate(tscv.split(returns_series)):
        fold_returns = returns_series.iloc[val_idx]
        fold_sharpes = empyrical.sharpe_ratio(fold_returns, period='daily')
        all_folds_performance.append({'trade_time':fold_returns.index[-1],'result':fold_sharpes})
    df = pd.DataFrame(all_folds_performance).set_index('trade_time')['result']
    return df