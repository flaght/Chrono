### level 绩效因子
import pdb, os, argparse, itertools, re, pdb
import pandas as pd
import numpy as np
from pymongo import InsertOne, DeleteOne
from dotenv import load_dotenv

load_dotenv()
from ultron.factor.genetic.geneticist.operators import calc_factor
from ultron.sentry.api import *
from kdutils.mongodb import MongoDBManager
from kdutils.common import fetch_temp_data, fetch_temp_returns
from lumina.genetic.metrics.evaluate import FactorEvaluate
from lumina.genetic.process import *


# 四则运算
def add(x, y):
    return np.add(x, y)


def sub(x, y):
    return np.subtract(x, y)


def mul(x, y):
    return np.multiply(x, y)


def div(x, y):
    return np.divide(x, y)


def neg(x):
    return np.negative(x)


def abs(x):
    return np.fabs(x)


def sqrt(x):
    return np.sqrt(x)


# 简单算子
def mavg(series, window):
    return series.rolling(window).mean()


def msum(series, window):
    return series.rolling(window).sum()


def mcount(series, window):
    return series.rolling(window).count()


def mprod(series, window):
    # 使用对数求和避免大数溢出
    return np.exp(
        series.rolling(window).apply(lambda x: np.sum(np.log(x)), raw=True))


def mvar(series, window):
    return series.rolling(window).var(ddof=1)


def mstd(series, window):
    return series.rolling(window).std(ddof=1)


def mskew(series, window):
    return series.rolling(window).skew()


def mkurtosis(series, window):
    return series.rolling(window).kurt()


def mmin(series, window):
    return series.rolling(window).min()


def mmax(series, window):
    return series.rolling(window).max()


def mimin(series, window):
    return series.rolling(window).apply(np.argmin, raw=True)


def mimax(series, window):
    return series.rolling(window).apply(np.argmax, raw=True)


def mmed(series, window):
    return series.rolling(window).median()


def mmad(series, window):
    return series.rolling(window).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)


def mrank(series, window):
    return series.rolling(window).rank(pct=True)


def mfirst(series, window):
    return series.rolling(window).apply(lambda x: x[0], raw=True)


def mlast(series, window):
    return series.rolling(window).apply(lambda x: x[-1], raw=True)


def mmaxPositiveStreak(series, window):
    # 优化连续正数和计算
    def max_positive_streak(x):
        pos = np.where(x > 0, x, 0)
        cumsum = pos.cumsum()
        reset = np.where(pos == 0)[0]
        if reset.size > 0:
            cumsum -= np.repeat(cumsum[reset], np.diff(np.append(-1, reset)))
        return np.max(cumsum)

    return series.rolling(window).apply(max_positive_streak, raw=True)


# ================= 单目无参算子（向量化实现）=================
def deltas(series):
    return series.diff()


def ratios(series):
    return series / series.shift(1)


def prev(series):
    return series.shift(1)


def percentChange(series):
    return series.pct_change()


# ================= 双目算子（并行化优化）=================
def mcorr(series1, series2, window):
    # 直接使用滚动相关系数
    return series1.rolling(window).corr(series2)


def mcovar(series1, series2, window):
    # 直接使用滚动协方差
    return series1.rolling(window).cov(series2)


def mbeta(y, x, window):
    # 向量化beta系数计算
    cov = y.rolling(window).cov(x)
    var_x = x.rolling(window).var()
    return cov / var_x.replace(0, np.nan)


def mwsum(x, y, window):
    # 向量化加权和计算
    return (x * y).rolling(window).sum()


def mwavg(x, weights, window):
    # 向量化加权平均
    weighted_sum = (x * weights).rolling(window).sum()
    total_weight = weights.rolling(window).sum()
    return weighted_sum / total_weight.replace(0, np.nan)


# ================= TA-Lib时序算子（优化实现）=================
def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def dema(series, window):
    ema1 = series.ewm(span=window, adjust=False).mean()
    return 2 * ema1 - ema1.ewm(span=window, adjust=False).mean()


def tema(series, window):
    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema2.ewm(span=window, adjust=False).mean()


def trima(series, window):
    # 双重平滑优化
    half_window = (window + 1) // 2
    return series.rolling(half_window).mean().rolling(half_window).mean()


def wma(series, window):
    # 预计算权重向量
    weights = np.arange(1, window + 1)
    total_weight = weights.sum()
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights) / total_weight, raw=True)


def rsi(series, window):
    # 向量化RSI计算
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_expression(expression_str: str,
                         data_df: pd.DataFrame) -> pd.Series:
    """
    动态解析并计算一个字符串表达式。

    参数:
    expression_str: 像 "mrank(mskew(neg(column_name), 192), 192)" 这样的字符串。
    data_df: 包含所有列数据的 Pandas DataFrame。

    返回:
    计算结果，一个 Pandas Series。
    """

    # 1. 定义一个安全的函数映射，只有这里的函数才允许被执行。
    allowed_functions = {
        # 四则运算
        'add': add,
        'sub': sub,
        'mul': mul,
        'div': div,
        'neg': neg,
        'abs': abs,
        'sqrt': sqrt,

        # 简单算子 (滚动窗口类)
        'mavg': mavg,
        'msum': msum,
        'mcount': mcount,
        'mprod': mprod,
        'mvar': mvar,
        'mstd': mstd,
        'mskew': mskew,
        'mkurtosis': mkurtosis,
        'mmin': mmin,
        'mmax': mmax,
        'mimin': mimin,
        'mimax': mimax,
        'mmed': mmed,
        'mmad': mmad,
        'mrank': mrank,
        'mfirst': mfirst,
        'mlast': mlast,
        'mmaxPositiveStreak': mmaxPositiveStreak,

        # 单目无参算子
        'deltas': deltas,
        'ratios': ratios,
        'prev': prev,
        'percentChange': percentChange,

        # 双目算子
        'mcorr': mcorr,
        'mcovar': mcovar,
        'mbeta': mbeta,
        'mwsum': mwsum,
        'mwavg': mwavg,

        # TA-Lib时序算子
        'ema': ema,
        'dema': dema,
        'tema': tema,
        'trima': trima,
        'wma': wma,
        'rsi': rsi
    }

    # 2. 准备 eval 的执行上下文环境 (locals)
    eval_context = {}

    # 将允许的函数添加到上下文中
    eval_context.update(allowed_functions)

    # 3. 从表达式中提取所有可能的变量名（列名）
    #    这个正则表达式会找到所有合法的Python标识符
    potential_vars = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expression_str)

    # 4. 遍历这些变量，如果它们是DataFrame的列名，就将对应的Series添加到上下文中
    for var in set(potential_vars):
        if var in data_df.columns:
            eval_context[var] = data_df[var]

    # 5. 使用 eval 执行计算
    #    globals 参数设为空字典 {} 以增强安全性，防止访问全局变量。
    #    locals 参数是我们精心构建的、只包含安全函数和所需数据列的上下文。
    try:
        result = eval(expression_str, {"__builtins__": {}}, eval_context)
        if isinstance(result, pd.Series):
            return result
        else:
            # 如果结果不是Series，可能表达式有问题，比如 "1+1"
            raise TypeError(
                f"Expression did not result in a Pandas Series. Result: {result}"
            )
    except Exception as e:
        print(f"Error evaluating expression: '{expression_str}'")
        print(f"Error details: {e}")
        # 可以在这里返回 None 或者重新抛出异常
        raise


@add_process_env_sig
def callback_fitness(target_column, method, instruments, fee, total_data,
                     returns_data):
    res = []
    for column in target_column:
        print(column)
        try:
            results = calc_fitness(target_column=column,
                                   method=method,
                                   instruments=instruments,
                                   fee=fee,
                                   total_data=total_data,
                                   returns_data=returns_data)
            res.append(results)
        except Exception as e:
            print("error:{0}:{1}".format(column, str(e)))
    return res


def calc_fitness(target_column, method, instruments, fee, total_data,
                 returns_data):
    columns = target_column['columns']
    params = target_column['params']
    factors_data1 = total_data[[
        columns, 'code'
    ]].reset_index().rename(columns={columns: 'transformed'})
    total_data1 = factors_data1.merge(returns_data, on=['trade_time',
                                                        'code']).dropna()
    res = []
    for param in params:
        scale_method = param['scale_method']
        roll_win = param['roll_win']
        ret_name = param['ret_name']
        MyFactorBacktest = FactorEvaluate(
            factor_data=total_data1,
            factor_name='transformed',
            ret_name=ret_name,
            roll_win=roll_win,  # 因子放缩窗口，自定义
            fee=fee,
            scale_method=scale_method)  # 可换 'roll_zscore' 等

        result = MyFactorBacktest.run()
        result['name'] = target_column['columns']
        result['scale_method'] = scale_method
        result['roll_win'] = roll_win
        result['ret_name'] = ret_name
        result['method'] = method
        result['instruments'] = instruments
        res.append(pd.DataFrame([result]))
    return res


## 文件读取 若需要刷新 则从另外函数执行
def fetch_market(instruments, method, task_id, name):
    total_data = fetch_temp_data(
        method=method,
        task_id=task_id,
        instruments=instruments,
        datasets=name if isinstance(name, list) else [name])

    total_returns = fetch_temp_returns(
        method=method,
        task_id=task_id,
        instruments=instruments,
        datasets=name if isinstance(name, list) else [name],
        category='returns')
    return total_data, total_returns


## 批量计算因子
def batch(total_data):
    res = []
    individual = pd.read_csv("records/individual_factor.csv",index_col=0)
    individual = individual['individual'].tolist()
    for ind in individual[:20]:
        result = calculate_expression(expression_str=ind, data_df=total_data)
        result = result.dropna()
        result.name = ind
        if not result.empty:
            res.append(result)
    return pd.concat(res, axis=1)


def run(method, instruments, task_id):
    total_data, total_returns = fetch_market(instruments=instruments,
                                             method=method,
                                             task_id=task_id,
                                             name=['train', 'val', 'test'])
    total_data = total_data.sort_values(by=['trade_time', 'code'])
    total_returns = total_returns.sort_values(by=['trade_time', 'code'])
    total_data = total_data.set_index('trade_time')
    scale_method_sets = [
        'roll_min_max',
        'roll_zscore',
        'roll_quantile',
        'ew_zscore'  #,'train_const'
    ]
    returns_columns = ['time_weight', 'equal_weight']
    roll_win_sets = [60, 120, 240, 300]

    params_sets = [
        {
            'scale_method': scale_method,
            'roll_win': roll_win,
            'ret_name': return_col
        } for scale_method in scale_method_sets  # 遍历每个放缩方法
        for roll_win in roll_win_sets  # 遍历每个滚动窗口
        for return_col in returns_columns
    ]
    res = [
        {
            'columns': columns,
            'params': params_sets,
        } for columns in factors_set  # 遍历每个原始表达式字典
    ]

    k_split = 4
    process_list = split_k(k_split, res)
    results = create_parellel(process_list=process_list,
                              callback=callback_fitness,
                              method=method,
                              fee=0.0000005,
                              instruments=instruments,
                              total_data=total_data,
                              returns_data=total_returns)
    pdb.set_trace()
    res1 = list(itertools.chain.from_iterable(results))
    res1 = [rs for res in res1 for rs in res]
    results = pd.concat(res1)
    print('-->')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--method',
                        type=str,
                        default='aicso0',
                        help='data method')
    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instruments')

    parser.add_argument('--task_id',
                        type=str,
                        default='200037',
                        help='code or instruments')

    args = parser.parse_args()
    run(method=args.method, instruments=args.instruments, task_id=args.task_id)
