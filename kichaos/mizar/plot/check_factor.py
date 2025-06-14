from scipy.stats import skew
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.stats import kurtosis, boxcox  # boxcox 用于演示转换，尽管对峰度效果有限
from scipy.stats.mstats import winsorize
from alphacopilot.calendars.api import *
from kdutils.macro2 import *


def excess_kurtosis_func(data_series, feature_name="Feature", name='train'):
    """
    计算数据序列的超额峰度，并根据经验法则进行评估。
    """
    ek = kurtosis(data_series.dropna(), fisher=True, bias=True)
    print(f"超额峰度 (Excess Kurtosis): {ek:.4f}")

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

    print(f"{name} 评估: {interpretation}")

    if ek > 1:
        print(f"{name}  提醒: 厚尾意味着极端值出现的概率高于正态分布预期。这对风险管理和某些依赖正态性假设的模型很重要。")
    elif ek < -0.5:  # 例如，比-0.5更负
        print(f"{name}  提醒: 薄尾意味着极端值出现的概率低于正态分布预期。数据可能比正态分布更均匀或有界。")
    return ek, interpretation


def skewness_func(data_series, feature_name="Feature"):
    """
    计算数据序列的偏度，并根据经验法则进行评估。
    """
    s = skew(data_series.dropna())  # dropna以防有缺失值影响计算

    print(f"\n--- {feature_name} 偏度分析 ---")
    print(f"偏度 (Skewness): {s:.4f}")

    abs_s = abs(s)
    if abs_s < 0.5:
        interpretation = "分布近似对称 (Approx. Symmetrical)"
    elif 0.5 <= abs_s < 1:
        interpretation = "分布中等程度偏斜 (Moderately Skewed)"
    else:  # abs_s >= 1
        interpretation = "分布高度偏斜 (Highly Skewed)"

    if s > 0.05:  # 加一个小阈值避免非常接近0的也被判为正偏
        direction = "正偏/右偏 (Right-skewed)"
    elif s < -0.05:
        direction = "负偏/左偏 (Left-skewed)"
    else:
        direction = "接近对称 (Nearly Symmetrical)"

    print(f"评估: {interpretation}")
    if interpretation != "分布近似对称 (Approx. Symmetrical)":
        print(f"偏斜方向: {direction}")

    # 提醒潜在的模型影响
    if abs_s >= 1:
        print("提醒: 高度偏斜可能影响某些依赖数据对称性的模型 (如线性回归的OLS假设)。")
    elif 0.5 <= abs_s < 1:
        print("提醒: 中等偏斜，根据模型需求，可能需要考虑数据转换。")

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
    print(f"{name} 均值: {mean_val:.4f}")
    print(
        f"{name} 均值的 {confidence_level*100:.0f}% 置信区间: [{mean_ci_lower:.4f}, {mean_ci_upper:.4f}]"
    )

    mean_ci_width = mean_ci_upper - mean_ci_lower
    relative_width_mean = mean_ci_width / abs(mean_val)  # 使用绝对值，因为均值可能为负
    print(f"{name} 均值置信区间的绝对宽度: {mean_ci_width:.4f}")
    print("\n\n")


def overall_mean(train1, val1, test1, factor_name):
    train_mean = train1[factor_name].mean()
    val_mean = val1[factor_name].mean()
    test_mean = test1[factor_name].mean()
    print("train_mean:{0}\nval_mean:{1}\ntest_mean:{2}".format(
        train_mean, val_mean, test_mean))


def rolling_mean(train1, val1, test1, factor_name, window):
    pos = len(test1) - window
    train_mean = train1[factor_name].rolling(window).mean()
    val_mean = val1[factor_name].rolling(window).mean()
    test_mean = test1[factor_name].rolling(window).mean()

    train_mean.name = 'train'
    val_mean.name = 'val'
    test_mean.name = 'test'
    print("rolling window:{0}".format(window))
    print("""
      train mean  max:{0},min:{1},mean:{2},std:{3}
      val mean  max:{4},min:{5},mean:{6},std:{7}
      test mean  max:{8},min:{9},mean:{10},std:{11}
      """.format(train_mean.max(), train_mean.min(), train_mean.mean(),
                 train_mean.std(), val_mean.max(), val_mean.min(),
                 val_mean.mean(), val_mean.std(), test_mean.max(),
                 test_mean.min(), test_mean.mean(), test_mean.std()))

    pd.concat([
        train_mean[-pos:].reset_index(drop=True), val_mean[-pos:].reset_index(
            drop=True), test_mean[-pos:].reset_index(drop=True)
    ],
              axis=1).plot(figsize=(6, 3))


def confidence_interval(train1, val1, test1, factor_name):
    confidence_func(train1[factor_name], name="train")
    confidence_func(val1[factor_name], name="val")
    confidence_func(test1[factor_name], name="test")


def evaluate_skewness(train1, val1, test1, factor_name):
    s_right_train, _, _ = skewness_func(train1[factor_name], factor_name)
    s_right_val, _, _ = skewness_func(val1[factor_name], factor_name)
    s_right_test, _, _ = skewness_func(test1[factor_name], factor_name)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    sns.histplot(train1[factor_name], kde=True, ax=axes[0])
    axes[0].set_title(f'train data\nSkewness: {s_right_train:.2f}')
    sns.histplot(val1[factor_name], kde=True, ax=axes[1])
    axes[1].set_title(f'val data \nSkewness: {s_right_val:.2f}')

    sns.histplot(test1[factor_name], kde=True, ax=axes[2])
    axes[2].set_title(f'test data \nSkewness: {s_right_test:.2f}')

    plt.show()


def evaluate_excess_kurtosis(train1, val1, test1, factor_name):
    ek_original_train, interpretation_original_train = excess_kurtosis_func(
        train1[factor_name], train1[factor_name].name, 'train')
    ek_original_val, interpretation_original_val = excess_kurtosis_func(
        val1[factor_name], val1[factor_name].name, 'val')
    ek_original_test, interpretation_original_test = excess_kurtosis_func(
        test1[factor_name], test1[factor_name].name, 'test')
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    sns.histplot(train1[factor_name], kde=True, bins=100,
                 ax=axes[0])  # 增加bins数量以更好观察20万数据的分布
    axes[0].set_title(f'train Excess Kurtosis: {ek_original_train:.2f}')

    sns.histplot(val1[factor_name], kde=True, bins=100,
                 ax=axes[1])  # 增加bins数量以更好观察20万数据的分布
    axes[1].set_title(f'val Excess Kurtosis: {ek_original_val:.2f}')

    sns.histplot(test1[factor_name], kde=True, bins=100,
                 ax=axes[2])  # 增加bins数量以更好观察20万数据的分布
    axes[2].set_title(f'test Excess Kurtosis: {ek_original_val:.2f}')

    plt.show()


def simple_factors(train1, val1, test1, factor_name):
    overall_mean(train1=train1,
                 val1=val1,
                 test1=test1,
                 factor_name=factor_name)

    rolling_mean(train1=train1,
                 val1=val1,
                 test1=test1,
                 factor_name=factor_name,
                 window=40)

    rolling_mean(train1=train1,
                 val1=val1,
                 test1=test1,
                 factor_name=factor_name,
                 window=80)

    confidence_interval(train1=train1,
                        val1=val1,
                        test1=test1,
                        factor_name=factor_name)

    evaluate_skewness(train1=train1,
                      val1=val1,
                      test1=test1,
                      factor_name=factor_name)

    evaluate_excess_kurtosis(train1=train1,
                             val1=val1,
                             test1=test1,
                             factor_name=factor_name)
