import pandas as pd

FLOAT_EPS = 1e-8


def safe_quantile(series: pd.Series, q: float) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.quantile(q))


def safe_mean(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def safe_median(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.median())


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(a.corr(b, method=method))


def safe_base(value, floor=1e-8):
    return max(abs(value), floor)