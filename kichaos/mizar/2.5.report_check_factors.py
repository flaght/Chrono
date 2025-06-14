import os, datetime
import pandas as pd
import numpy as np
import pdb, argparse
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import *
from kdutils.macro2 import *
#from plot.check_factor import *


def load_data(variant):
    dirs = os.path.join(
        base_path, 'mizar', variant['method'], 'normal',
        variant['g_instruments'], 'rolling', 'normal_factors3',
        "{0}_{1}".format(variant['types'], variant['horizon']),
        "{0}_{1}_{2}_{3}_{4}".format(str(variant['freq']),
                                     str(variant['train_days']),
                                     str(variant['val_days']),
                                     str(variant['nc']),
                                     str(variant['window'])))

    train1 = pd.read_feather(
        os.path.join(
            dirs, "normal_factors_train_{0}.feather".format(variant['index'])))
    val1 = pd.read_feather(
        os.path.join(dirs, "normal_factors_val_{0}.feather".format(
            variant['index'])))
    test1 = pd.read_feather(
        os.path.join(
            dirs, "normal_factors_test_{0}.feather".format(variant['index'])))

    return train1, val1, test1


def excess_kurtosis_func(data_series, feature_name="Feature", name='train'):
    """
    计算数据序列的超额峰度，并根据经验法则进行评估。
    """
    ek = kurtosis(data_series.dropna(), fisher=True, bias=True)
    #print(f"超额峰度 (Excess Kurtosis): {ek:.4f}")

    abs_ek = abs(ek)
    if abs_ek < 1:
        interpretation = "尾部行为与正态分布无显著差异 (Mesokurtic-like)"
    elif ek > 0 and 1 <= ek < 3:
        interpretation = "中等程度尖峰/厚尾 (Moderately Leptokurtic)"
    elif ek > 0 and ek >= 3:
        interpretation = "高度尖峰/厚尾 (Highly Leptokurtic / Fat-tailed)"
    elif ek < 0 and -3 < ek <= -1:  # 注意这里阈值与之前略有调整，使其更连续
        interpretation = "中等程度平顶/薄尾 (Moderately Platykurtic)"
    elif ek < 0 and ek <= -3:
        interpretation = "高度平顶/薄尾 (Highly Platykurtic)"
    else:  # 比如 ek 在 -1 到 0 之间
        interpretation = "轻微平顶/薄尾或接近正态 (Slightly Platykurtic or near Mesokurtic)"

    #print(f"{name} 评估: {interpretation}")

    #if ek > 1:
    #    print(f"{name}  提醒: 厚尾意味着极端值出现的概率高于正态分布预期。这对风险管理和某些依赖正态性假设的模型很重要。")
    #elif ek < -0.5:  # 例如，比-0.5更负
    #    print(f"{name}  提醒: 薄尾意味着极端值出现的概率低于正态分布预期。数据可能比正态分布更均匀或有界。")
    return ek, interpretation


def skewness_func(data_series, feature_name="Feature"):
    """
    计算数据序列的偏度，并根据经验法则进行评估。
    """
    s = skew(data_series.dropna())  # dropna以防有缺失值影响计算

    #print(f"\n--- {feature_name} 偏度分析 ---")
    #print(f"偏度 (Skewness): {s:.4f}")

    abs_s = abs(s)
    if abs_s < 0.5:
        interpretation = "分布近似对称 (Approx. Symmetrical)"
    elif 0.5 <= abs_s < 1:
        interpretation = "分布中等程度偏斜 (Moderately Skewed)"
    else:  # abs_s >= 1
        interpretation = "分布高度偏斜 (Highly Skewed)"

    if s > 0.05 and interpretation != "分布近似对称 (Approx. Symmetrical)":  # 加一个小阈值避免非常接近0的也被判为正偏
        direction = "正偏/右偏 (Right-skewed)"
    elif s < -0.05 and interpretation != "分布近似对称 (Approx. Symmetrical)":
        direction = "负偏/左偏 (Left-skewed)"
    else:
        direction = "接近对称 (Nearly Symmetrical)"

    #print(f"评估: {interpretation}")
    #if interpretation != "分布近似对称 (Approx. Symmetrical)":
    #    print(f"偏斜方向: {direction}")

    # 提醒潜在的模型影响
    #if abs_s >= 1:
    #    print("提醒: 高度偏斜可能影响某些依赖数据对称性的模型 (如线性回归的OLS假设)。")
    #elif 0.5 <= abs_s < 1:
    #    print("提醒: 中等偏斜，根据模型需求，可能需要考虑数据转换。")

    return s, interpretation, direction


def confidence_func(data, confidence_level=0.95, name="训练集"):
    #### 置信区间
    mean_val = data.mean()
    std_err_mean = data.sem()  # 标准误差

    ## 需要确保 std_err_mean 大于0 即标准误差有效
    mean_ci_lower, mean_ci_upper = stats.t.interval(confidence_level,
                                                    len(data) - 1,
                                                    loc=mean_val,
                                                    scale=std_err_mean)
    #print(f"{name} 均值: {mean_val:.4f}")
    #print(
    #    f"{name} 均值的 {confidence_level*100:.0f}% 置信区间: [{mean_ci_lower:.4f}, {mean_ci_upper:.4f}]"
    #)

    mean_ci_width = mean_ci_upper - mean_ci_lower
    return confidence_level, mean_ci_lower, mean_ci_upper, mean_ci_width


def overall_mean(train1, val1, test1, factor_name, report):
    report['overall_train_mean'] = train1[factor_name].mean()
    report['overall_val_mean'] = val1[factor_name].mean()
    report['overall_test_mean'] = test1[factor_name].mean()
    return report


def rolling_mean(train1, val1, test1, factor_name, window, report):
    train_mean = train1[factor_name].rolling(window).mean()
    val_mean = val1[factor_name].rolling(window).mean()
    test_mean = test1[factor_name].rolling(window).mean()

    report[f'{window}w_train_rolling_max'] = train_mean.max()
    report[f'{window}w_train_rolling_min'] = train_mean.min()
    report[f'{window}w_train_rolling_mean'] = train_mean.mean()
    report[f'{window}w_train_rolling_std'] = train_mean.std()

    report[f'{window}w_val_rolling_max'] = val_mean.max()
    report[f'{window}w_val_rolling_min'] = val_mean.min()
    report[f'{window}w_val_rolling_mean'] = val_mean.mean()
    report[f'{window}w_val_rolling_std'] = val_mean.std()

    report[f'{window}w_test_rolling_max'] = test_mean.max()
    report[f'{window}w_test_rolling_min'] = test_mean.min()
    report[f'{window}w_test_rolling_mean'] = test_mean.mean()
    report[f'{window}w_test_rolling_std'] = test_mean.std()
    return report


def confidence_interval(train1, val1, test1, factor_name, report):
    train_confidence_level, train_mean_ci_lower, train_mean_ci_upper, train_mean_ci_width = confidence_func(
        train1[factor_name], name="train")
    val_confidence_level, val_mean_ci_lower, val_mean_ci_upper, val_mean_ci_width = confidence_func(
        val1[factor_name], name="val")
    test_confidence_level, test_mean_ci_lower, test_mean_ci_upper, test_mean_ci_width = confidence_func(
        test1[factor_name], name="test")
    report['train_confidence_level'] = train_confidence_level
    report['train_mean_ci_lower'] = train_mean_ci_lower
    report['train_mean_ci_upper'] = train_mean_ci_upper
    report['train_mean_ci_width'] = train_mean_ci_width

    report['val_confidence_level'] = val_confidence_level
    report['val_mean_ci_lower'] = val_mean_ci_lower
    report['val_mean_ci_upper'] = val_mean_ci_upper
    report['val_mean_ci_width'] = val_mean_ci_width

    report['test_confidence_level'] = test_confidence_level
    report['test_mean_ci_lower'] = test_mean_ci_lower
    report['test_mean_ci_upper'] = test_mean_ci_upper
    report['test_mean_ci_width'] = test_mean_ci_width


def evaluate_skewness(train1, val1, test1, factor_name, report):
    s_right_train, interpretation_train, direction_train = skewness_func(
        train1[factor_name], factor_name)
    s_right_val, interpretation_val, direction_val = skewness_func(
        val1[factor_name], factor_name)
    s_right_test, interpretation_test, direction_test = skewness_func(
        test1[factor_name], factor_name)

    report['s_right_train'] = s_right_train
    report['interpretation_train'] = interpretation_train
    report['direction_train'] = direction_train

    report['s_right_val'] = s_right_val
    report['interpretation_val'] = interpretation_val
    report['direction_val'] = direction_val

    report['s_right_test'] = s_right_test
    report['interpretation_test'] = interpretation_test
    report['direction_test'] = direction_test


def evaluate_excess_kurtosis(train1, val1, test1, factor_name, report):
    ek_original_train, interpretation_original_train = excess_kurtosis_func(
        train1[factor_name], train1[factor_name].name, 'train')
    ek_original_val, interpretation_original_val = excess_kurtosis_func(
        val1[factor_name], val1[factor_name].name, 'val')
    ek_original_test, interpretation_original_test = excess_kurtosis_func(
        test1[factor_name], test1[factor_name].name, 'test')

    report['ek_original_train'] = ek_original_train
    report['interpretation_original_train'] = interpretation_original_train

    report['ek_original_val'] = ek_original_val
    report['interpretation_original_val'] = interpretation_original_val

    report['ek_original_test'] = ek_original_test
    report['interpretation_original_test'] = interpretation_original_test


# 使用IQR（四分位距）方法识别并计算异常值的百分比。
## 异常值被定义为小于 Q1 - 1.5*IQR 或大于 Q3 + 1.5*IQR 的数据点。
def outliers_iqr_func(data_series, feature_name="Feature", name='train'):
    # 丢弃缺失值，避免影响计算
    clean_data = data_series.dropna()
    # 如果数据量太少，无法计算分位数，则返回0
    if clean_data.empty:
        return 0.0

    # 计算第一四分位数 (Q1) 和第三四分位数 (Q3)
    q1 = clean_data.quantile(0.25)
    q3 = clean_data.quantile(0.75)

    # 计算四分位距 (IQR)
    iqr = q3 - q1

    # 定义异常值的边界
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 识别出界的数据点
    outliers = clean_data[(clean_data < lower_bound) |
                          (clean_data > upper_bound)]

    # 计算异常值百分比
    #outlier_percentage = len(outliers) * 100 / len(clean_data)
    outlier_percentage = len(outliers) / len(clean_data)
    return outlier_percentage


def evaluate_outliers_iqr(train1, val1, test1, factor_name, report):
    train_outlier_percentage = outliers_iqr_func(train1[factor_name],
                                                 train1[factor_name].name,
                                                 'train')
    val_outlier_percentage = outliers_iqr_func(val1[factor_name],
                                               val1[factor_name].name, 'val')
    test_outlier_percentage = outliers_iqr_func(test1[factor_name],
                                                val1[factor_name].name, 'test')

    report['train_outlier_percent'] = train_outlier_percentage
    report['val_outlier_percent'] = val_outlier_percentage
    report['test_outlier_percent'] = test_outlier_percentage
    return report


# 使用增强迪基-福勒（ADF）检验来检查时间序列的平稳性
def stationarity_adf_func(data_series,
                          significance_level=0.05,
                          feature_name="Feature",
                          name='train'):
    # ADF检验无法处理NaN值，先丢弃
    clean_data = data_series.dropna()
    # ADF检验需要足够的数据点才能得出有意义的结果
    if len(clean_data) < 20:
        print(
            "Warning: Data series has fewer than 20 points after dropping NaNs. ADF test skipped."
        )
        return None, None

    # 执行ADF检验
    # autolag='AIC' 会自动选择最佳的滞后阶数
    result = adfuller(clean_data, maxlag=10) # maxlag=10 autolag='AIC'
    p_value = result[1]

    # ADF检验的原假设是：序列存在单位根（即非平稳）。
    # 如果p值小于显著性水平，我们拒绝原假设，认为序列是平稳的。
    is_stationary = p_value < significance_level
    return p_value, is_stationary


def evaluate_stationarity_adf(train1, val1, test1, factor_name, report):
    train_p_value, train_is_stationary = stationarity_adf_func(
        data_series=train1[factor_name],
        feature_name=train1[factor_name].name,
        name='train')

    val_p_value, val_is_stationary = stationarity_adf_func(
        data_series=val1[factor_name],
        feature_name=val1[factor_name].name,
        name='val')

    test_p_value, test_is_stationary = stationarity_adf_func(
        data_series=test1[factor_name],
        feature_name=test1[factor_name].name,
        name='test')

    report['train_p_value'] = train_p_value
    report['train_is_stationary'] = train_is_stationary

    report['val_p_value'] = val_p_value
    report['val_is_stationary'] = val_is_stationary

    report['test_p_value'] = test_p_value
    report['test_is_stationary'] = test_is_stationary

    return report


def simple_check(variant):
    train1, val1, test1 = load_data(variant=variant)

    features = [
        col for col in train1.columns
        if col not in ['trade_time', 'code', 'price']
    ]
    reports = []
    for factor_name in features:
        print(factor_name)
        report = {'name': factor_name}
        ### 整体均值
        overall_mean(train1=train1,
                     val1=val1,
                     test1=test1,
                     factor_name=factor_name,
                     report=report)

        ### 滚动 均值
        rolling_mean(train1=train1,
                     val1=val1,
                     test1=test1,
                     factor_name=factor_name,
                     window=40,
                     report=report)

        rolling_mean(train1=train1,
                     val1=val1,
                     test1=test1,
                     factor_name=factor_name,
                     window=60,
                     report=report)

        ### 置信度
        confidence_interval(train1=train1,
                            val1=val1,
                            test1=test1,
                            factor_name=factor_name,
                            report=report)

        ### 偏度
        evaluate_skewness(train1=train1,
                          val1=val1,
                          test1=test1,
                          factor_name=factor_name,
                          report=report)

        ### 峰度
        evaluate_excess_kurtosis(train1=train1,
                                 val1=val1,
                                 test1=test1,
                                 factor_name=factor_name,
                                 report=report)

        ### IQR 异常值
        evaluate_outliers_iqr(train1=train1,
                              val1=val1,
                              test1=test1,
                              factor_name=factor_name,
                              report=report)
        ### 时间序列的平稳性
        evaluate_stationarity_adf(train1=train1,
                                  val1=val1,
                                  test1=test1,
                                  factor_name=factor_name,
                                  report=report)

        reports.append(report)

    dirs = os.path.join(
        base_path, 'mizar', variant['method'], 'normal',
        variant['g_instruments'], 'rolling', 'normal_factors3',
        "{0}_{1}".format(variant['types'], variant['horizon']),
        "{0}_{1}_{2}_{3}_{4}".format(str(variant['freq']),
                                     str(variant['train_days']),
                                     str(variant['val_days']),
                                     str(variant['nc']),
                                     str(variant['window'])))
    reports = pd.DataFrame(reports)
    filename = os.path.join(
        dirs, "normal_factors_reports_{0}.csv".format(variant['index']))
    print("output:=======>{0}".format(filename))
    pdb.set_trace()
    reports.to_csv(filename, encoding="UTF-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--freq', type=int, default=10, help='symbol')

    parser.add_argument('--g_instruments',
                        type=str,
                        default='rbb',
                        help='symbol')

    parser.add_argument('--train_days',
                        type=int,
                        default=180,
                        help='Training days')  ## 训练天数

    parser.add_argument('--val_days',
                        type=int,
                        default=10,
                        help='Validation days')  ## 验证天数

    parser.add_argument('--category', type=int, default=1)

    
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--method',
                        type=str,
                        default='aicso4',
                        help='Method name')  ## 方法
    parser.add_argument('--types', type=str, default='o2o',
                        help='Types')  ## 类别
    parser.add_argument('--horizon',
                        type=int,
                        default=1,
                        help='Prediction horizon')  ## 预测周期

    parser.add_argument('--nc', type=int, default=2,
                        help='Standard method')  ## 标准方式

    parser.add_argument('--index', type=int, default=24, help='数据序列')  ## 标准方式

    args = parser.parse_args()

    variant = vars(args)
    simple_check(variant)
