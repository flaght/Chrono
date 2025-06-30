import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, boxcox, yeojohnson, t as t_dist, sem
from scipy.stats.mstats import winsorize
from dotenv import load_dotenv

load_dotenv()
from kdutils.macro2 import *
#import matplotlib.pyplot as plt
#import seaborn as sns
# --- 配置参数 ---
# 缺失值处理
NAN_THRESHOLD_DISCARD = 0.30
NAN_IMPUTE_METHOD = 'median'  # 'median' or 'mean'

# 标准差与变异系数处理
STD_DEV_THRESHOLD_DISCARD = 1e-6
CV_HIGH_THRESHOLD = 2.0
CV_VERY_LOW_MEAN_THRESHOLD = 1e-7  # 均值绝对值小于此值时，CV计算可能无意义或极大

# 偏度与峰度目标 (修正后的可接受范围)
SKEW_TARGET_MAX_ABS = 0.8  # 目标偏度绝对值上限 (可适当放宽)
KURTOSIS_TARGET_MAX_ABS = 2.0  # 目标超额峰度绝对值上限 (可适当放宽)

# Winsorization 参数
ENABLE_WINSORIZATION = True
WINSORIZE_LIMITS = (0.01, 0.01)  # (lower_quantile, upper_quantile)

# 形状转换参数
ENABLE_REFLECTION_FOR_LEFT_SKEW = True  # 是否为左偏数据启用反射 (用于log/sqrt/boxcox)

ENABLE_LOG_TRANSFORM = True  # 对数转换 (np.log1p)
LOG_TRANSFORM_MIN_SKEW_FOR_ATTEMPT = 0.75  # 当偏度大于此值(右偏)或小于负此值(左偏,需反射)时尝试

ENABLE_SQRT_TRANSFORM = False  # 平方根转换 (np.sqrt) - 通常效果弱于log，可选
SQRT_TRANSFORM_MIN_SKEW_FOR_ATTEMPT = 0.75

ENABLE_BOXCOX_TRANSFORM = True  # Box-Cox转换 (要求数据为正, 或反射后为正)
BOXCOX_MIN_ABS_SKEW_FOR_ATTEMPT = 0.75  # 尝试Box-Cox的偏度绝对值阈值

ENABLE_YEOJOHNSON_TRANSFORM = True  # Yeo-Johnson转换 (可处理正负和零)
YEOJOHNSON_MIN_ABS_SKEW_FOR_ATTEMPT = 0.75  # 尝试Yeo-Johnson的偏度绝对值阈值

# 置信区间
CONFIDENCE_LEVEL = 0.95
MEAN_CI_RELATIVE_WIDTH_THRESHOLD = 0.50

# 偏度/峰度评估文本阈值
SKEW_MODERATE_THRESHOLD = 0.5  # 用于判断是否左偏/右偏以进行反射
SKEW_HIGH_THRESHOLD = 1.0  # 用于最终评估解释
KURTOSIS_MODERATE_POSITIVE_THRESHOLD = 1.0
KURTOSIS_HIGH_POSITIVE_THRESHOLD = 3.0
KURTOSIS_MODERATE_NEGATIVE_THRESHOLD = -1.0


# --- 辅助函数 ---
def calculate_detailed_statistics(series,
                                  name="",
                                  confidence_level=CONFIDENCE_LEVEL):
    """计算详细统计特性，包括CV和均值置信区间"""
    if series.empty:
        return {
            'name':
            name,
            'count':
            0,
            'missing_pct':
            100.0 if series.isnull().all() and not series.empty else
            (series.isnull().mean() * 100 if not series.empty else 0),
            'mean':
            np.nan,
            'std_dev':
            np.nan,
            'cv':
            np.nan,
            'median':
            np.nan,
            'min':
            np.nan,
            'max':
            np.nan,
            'skewness':
            np.nan,
            'kurtosis (excess)':
            np.nan,
            'mean_ci_lower':
            np.nan,
            'mean_ci_upper':
            np.nan,
            'mean_ci_abs_width':
            np.nan,
            'mean_ci_relative_width':
            np.nan,
            'skew_interpretation':
            "N/A (empty series)",
            'kurtosis_interpretation':
            "N/A (empty series)",
            'mean_median_diff_norm_by_std':
            np.nan
        }

    series_numeric_original = pd.to_numeric(series, errors='coerce')
    series_numeric_dropped_na = series_numeric_original.dropna()

    stats = {
        'name': name,
        'count': series_numeric_dropped_na.count(),
        'missing_pct': series_numeric_original.isnull().mean() * 100,
        'mean': np.nan,
        'std_dev': np.nan,
        'cv': np.nan,
        'median': np.nan,
        'min': np.nan,
        'max': np.nan,
        'skewness': np.nan,
        'kurtosis (excess)': np.nan,
        'mean_ci_lower': np.nan,
        'mean_ci_upper': np.nan,
        'mean_ci_abs_width': np.nan,
        'mean_ci_relative_width': np.nan,
        'skew_interpretation': "N/A",
        'kurtosis_interpretation': "N/A",
        'mean_median_diff_norm_by_std': np.nan
    }

    if stats['count'] == 0:
        stats['skew_interpretation'] = "N/A (no valid data)"
        stats['kurtosis_interpretation'] = "N/A (no valid data)"
        # print(f"\n--- 详细统计特性: {name} (无有效数据) ---") # Suppressed for brevity in main loop
        # for key, value in stats.items(): print(f"{key.replace('_', ' ').capitalize():<30}: {value}")
        return stats

    mean_val = series_numeric_dropped_na.mean()
    std_val = series_numeric_dropped_na.std()
    median_val = series_numeric_dropped_na.median()
    current_skew_val = skew(series_numeric_dropped_na)
    current_kurt_val = kurtosis(
        series_numeric_dropped_na)  # scipy.stats.kurtosis is excess kurtosis

    stats.update({
        'mean': mean_val,
        'std_dev': std_val,
        'median': median_val,
        'min': series_numeric_dropped_na.min(),
        'max': series_numeric_dropped_na.max(),
        'skewness': current_skew_val,
        'kurtosis (excess)': current_kurt_val
    })

    if abs(
            mean_val
    ) < CV_VERY_LOW_MEAN_THRESHOLD:  # Avoid division by zero or near-zero
        stats[
            'cv'] = np.inf if std_val > 1e-9 else 0  # if std is also zero, cv is 0
    else:
        stats['cv'] = std_val / mean_val if mean_val != 0 else np.inf

    if std_val > 1e-9:  # Avoid division by zero if std is effectively zero
        stats['mean_median_diff_norm_by_std'] = (mean_val -
                                                 median_val) / std_val
    else:  # std is zero
        stats[
            'mean_median_diff_norm_by_std'] = 0 if mean_val == median_val else (
                np.inf if mean_val > median_val else -np.inf)

    if stats['count'] > 1 and std_val > 0:  # Need at least 2 points and some variance for CI
        std_err_mean = sem(series_numeric_dropped_na)
        if std_err_mean > 0:  # Ensure scale is positive for t.interval
            ci = t_dist.interval(confidence_level,
                                 stats['count'] - 1,
                                 loc=mean_val,
                                 scale=std_err_mean)
            stats['mean_ci_lower'] = ci[0]
            stats['mean_ci_upper'] = ci[1]
            stats['mean_ci_abs_width'] = ci[1] - ci[0]
            if abs(mean_val
                   ) > 1e-9:  # Avoid division by zero for relative width
                stats['mean_ci_relative_width'] = stats[
                    'mean_ci_abs_width'] / abs(mean_val)
            else:  # Mean is ~0, relative width can be huge or undefined
                stats['mean_ci_relative_width'] = np.inf if stats[
                    'mean_ci_abs_width'] > 1e-9 else 0
        else:  # No variance or SEM is zero, CI is just the mean
            stats['mean_ci_lower'] = mean_val
            stats['mean_ci_upper'] = mean_val
            stats['mean_ci_abs_width'] = 0
            stats['mean_ci_relative_width'] = 0

    s = stats['skewness']
    abs_s = abs(s)
    if np.isnan(s): stats['skew_interpretation'] = "N/A (skewness is NaN)"
    elif abs_s < SKEW_MODERATE_THRESHOLD: stats['skew_interpretation'] = "近似对称"
    elif SKEW_MODERATE_THRESHOLD <= abs_s < SKEW_HIGH_THRESHOLD:
        stats['skew_interpretation'] = "中等偏斜"
    else:
        stats['skew_interpretation'] = "高度偏斜"
    if not np.isnan(
            s
    ) and abs_s >= SKEW_MODERATE_THRESHOLD:  # Add direction for non-symmetrical
        stats['skew_interpretation'] += " (" + ("右偏" if s > 0 else "左偏") + ")"

    ek = stats['kurtosis (excess)']
    if np.isnan(ek): stats['kurtosis_interpretation'] = "N/A (kurtosis is NaN)"
    elif abs(
            ek
    ) < KURTOSIS_MODERATE_POSITIVE_THRESHOLD and ek >= KURTOSIS_MODERATE_NEGATIVE_THRESHOLD:
        stats['kurtosis_interpretation'] = "接近正态峰 (Mesokurtic-like)"
    elif ek >= KURTOSIS_HIGH_POSITIVE_THRESHOLD:
        stats['kurtosis_interpretation'] = "高度尖峰/厚尾 (Highly Leptokurtic)"
    elif ek >= KURTOSIS_MODERATE_POSITIVE_THRESHOLD:
        stats['kurtosis_interpretation'] = "中等尖峰/厚尾 (Moderately Leptokurtic)"
    elif ek <= KURTOSIS_MODERATE_NEGATIVE_THRESHOLD:  # e.g. < -1
        if ek < -2.0:  # Example threshold for highly platykurtic, adjust as needed
            stats['kurtosis_interpretation'] = "高度平顶/薄尾 (Highly Platykurtic)"
        else:
            stats[
                'kurtosis_interpretation'] = "中等平顶/薄尾 (Moderately Platykurtic)"
    else:  # e.g. ek between -1 and 0 for platykurtic side
        stats['kurtosis_interpretation'] = "轻微平顶/薄尾"

    # print(f"\n--- 详细统计特性: {name} ---") # Suppress printing from here for main loop
    # for key, value_stat in stats.items():
    #     if isinstance(value_stat, (int, float)) and not np.isnan(value_stat):
    #         print(f"{key.replace('_', ' ').capitalize():<30}: {value_stat:.4f}")
    #     else:
    #         print(f"{key.replace('_', ' ').capitalize():<30}: {value_stat}")
    return stats


def describe_factor_status(factor_name, status, reason=""):
    print(f"因子 '{factor_name}': {status}. {reason}")


def apply_transformation(series,
                         transform_type,
                         original_skew,
                         original_kurt,
                         enable_reflection_for_left_skew=True):
    """
    Helper to apply a transformation and check if it improved skew/kurtosis.
    Can optionally reflect left-skewed data before applying log/sqrt/boxcox.
    """
    transformed_series = series.copy()  # Work on a copy
    lambda_param = None
    applied_reflection = False
    reflection_constant_C = None  # Store C if reflection is applied

    # --- Reflection for left-skewed data if applicable ---
    # A factor is considered significantly left-skewed if its skewness is below a negative threshold
    is_left_skewed_significantly = original_skew < -SKEW_MODERATE_THRESHOLD

    # Only consider reflection for log1p, sqrt, boxcox when data is left-skewed and reflection is enabled
    needs_reflection_candidate = transform_type in ['log1p', 'sqrt', 'boxcox']

    if enable_reflection_for_left_skew and is_left_skewed_significantly and needs_reflection_candidate:
        # print(f"    左偏数据 (skew={original_skew:.2f}) for {transform_type}, 尝试反射...")
        # Use max/min of the current series (which might be post-winsorization)
        # Ensure series is not empty and has valid min/max
        if series.dropna().empty:
            return None, "数据为空无法反射", lambda_param, applied_reflection

        max_val = series.max()
        min_val = series.min()

        # Determine C for reflection X' = C - X
        # Goal: Make X' suitable for the chosen transform
        epsilon = 1e-6  # A small constant to ensure positivity or avoid boundary issues

        if transform_type == 'boxcox':  # X' must be > 0
            # C - X > 0  => C > X. So C must be > max_val.
            # A robust C to make all C-X positive:
            reflection_constant_C = max_val + abs(
                min_val) + epsilon if min_val < 0 else max_val + epsilon
            # Ensure C - min_val is indeed positive (smallest X' value)
            if reflection_constant_C - min_val <= 0:
                return None, "反射后仍无法确保所有值为正以应用Box-Cox", lambda_param, applied_reflection
        elif transform_type == 'sqrt':  # X' must be >= 0
            # C - X >= 0 => C >= X. So C must be >= max_val.
            reflection_constant_C = max_val  # C - max_val = 0 (acceptable for sqrt), C - min_val > 0
        elif transform_type == 'log1p':  # X' must be > -1
            # C - X > -1 => C > X - 1.
            # To be safe and simplify, let's aim for C - X > 0 (stricter than > -1), so C > max_val
            reflection_constant_C = max_val + epsilon
            if reflection_constant_C - min_val <= -1 + epsilon:  # Check C-min_val (smallest X') > -1
                return None, "反射后仍无法确保所有值 > -1 以应用log1p", lambda_param, applied_reflection

        transformed_series = reflection_constant_C - series  # Perform reflection
        applied_reflection = True
        # print(f"    反射完成 (C={reflection_constant_C:.4f}). 反射后数据临时偏度: {skew(transformed_series.dropna()):.2f}")

    # --- Apply actual transformation ---
    try:
        # Use data after potential reflection for these transforms
        # NaNs are handled by operating on .dropna() and then re-assigning to original series indices
        current_data_to_transform = transformed_series.dropna()
        if current_data_to_transform.empty:  # If all NaNs after reflection (or originally)
            return None, "无有效数据进行转换", lambda_param, applied_reflection

        if transform_type == 'log1p':
            if (current_data_to_transform
                    < -1 + epsilon).any():  # Check x > -1 strictly
                return None, "数据(或反射后)含<=-1的值,无法log1p", lambda_param, applied_reflection
            transformed_values = np.log1p(current_data_to_transform)
            transformed_series.loc[
                current_data_to_transform.index] = transformed_values
        elif transform_type == 'sqrt':
            if (current_data_to_transform < 0).any():
                return None, "数据(或反射后)含负值,无法sqrt", lambda_param, applied_reflection
            transformed_values = np.sqrt(current_data_to_transform)
            transformed_series.loc[
                current_data_to_transform.index] = transformed_values
        elif transform_type == 'boxcox':
            if not (current_data_to_transform
                    > 0).all():  # Must be all strictly positive
                return None, "数据(或反射后)不全为正,无法Box-Cox", lambda_param, applied_reflection
            if len(current_data_to_transform) < 2:
                return None, "数据点过少(<2),无法Box-Cox", lambda_param, applied_reflection
            transformed_values, lambda_param = boxcox(
                current_data_to_transform)
            transformed_series.loc[
                current_data_to_transform.index] = transformed_values
        elif transform_type == 'yeojohnson':
            # Yeo-Johnson handles signs and zero internally.
            # If reflection was applied before calling YJ (e.g. if YJ was tried after a reflected log attempt),
            # we should ideally operate on the original (pre-any-shape-transform) series.
            # The current logic passes `series_for_correction` which is post-winsor.
            # If `applied_reflection` is True from a previous attempt in the loop, YJ should ignore it.
            # So, for YJ, ensure we use the series state *before* any reflection specific to THIS transform call.
            if applied_reflection:  # If reflection was done in THIS call for YJ (which it shouldn't be for YJ)
                transformed_series = series.copy(
                )  # Reset to original series (pre-reflection for YJ)
                applied_reflection = False  # YJ does its own thing

            non_nan_data_for_yj = transformed_series.dropna(
            )  # Use original series (or post-winsor)
            if len(non_nan_data_for_yj) < 2:
                return None, "数据点过少(<2),无法Yeo-Johnson", lambda_param, applied_reflection
            transformed_values, lambda_param = yeojohnson(non_nan_data_for_yj)
            transformed_series.loc[
                non_nan_data_for_yj.index] = transformed_values
        else:
            return None, "未知的转换类型", lambda_param, applied_reflection

        new_skew = skew(transformed_series.dropna())
        new_kurt = kurtosis(transformed_series.dropna())

        # Improvement criteria: reduction in absolute skewness
        if abs(new_skew) < abs(original_skew):
            reflection_info = "(已反射)" if applied_reflection else ""
            return transformed_series, f"偏度从 {original_skew:.2f} 变为 {new_skew:.2f} {reflection_info}. 峰度从 {original_kurt:.2f} 变为 {new_kurt:.2f}.", lambda_param, applied_reflection
        else:
            reflection_info = "(已反射但未改善)" if applied_reflection else ""
            return None, f"转换{reflection_info}未改善偏度 (原abs:{abs(original_skew):.2f}, 新abs:{abs(new_skew):.2f}).", lambda_param, applied_reflection

    except Exception as e:
        return None, f"转换失败: {str(e)}", lambda_param, applied_reflection


# --- 主逻辑 ---
def screen_and_correct_factors(df, factor_cols):
    df_corrected = df.copy()
    report = {}

    for factor_name in factor_cols:
        print(f"\n{'='*15} 正在处理因子: {factor_name} {'='*15}")
        factor_report = {
            "actions": [],
            "final_status": "未知",
            "status_value": -1,
            "initial_stats_summary": None,
            "final_stats_summary": None,
            "warnings": [],
            "transformation_applied": "无"
        }

        current_series_orig = df_corrected[factor_name].copy()
        current_series = pd.to_numeric(current_series_orig, errors='coerce')

        print("计算初始统计...")
        initial_stats = calculate_detailed_statistics(current_series,
                                                      f"{factor_name} (原始)")
        factor_report["initial_stats_summary"] = {
            k: initial_stats[k]
            for k in [
                'mean', 'median', 'std_dev', 'cv', 'skewness',
                'kurtosis (excess)', 'missing_pct', 'count',
                'mean_ci_relative_width', 'skew_interpretation',
                'kurtosis_interpretation', 'mean_median_diff_norm_by_std'
            ]
        }
        # Print a concise summary
        print(
            f"初始统计: Count={initial_stats['count']}, Mean={initial_stats['mean']:.2f}, Median={initial_stats['median']:.2f}, "
            f"Std={initial_stats['std_dev']:.2f}, Skew={initial_stats['skewness']:.2f} ({initial_stats['skew_interpretation']}), "
            f"Kurt={initial_stats['kurtosis (excess)']:.2f} ({initial_stats['kurtosis_interpretation']}), "
            f"Missing={initial_stats['missing_pct']:.1f}%")

        if initial_stats['count'] == 0:
            message = f"所有值均为无效数值或缺失 (缺失比例: {initial_stats['missing_pct']:.2f}%)"
            factor_report["actions"].append(message)
            factor_report["final_status"] = f"抛弃 ({message})"
            factor_report['status_value'] = -1
            if factor_name in df_corrected.columns:
                df_corrected.drop(columns=[factor_name], inplace=True)
            describe_factor_status(factor_name, "抛弃", message)
            report[factor_name] = factor_report
            continue

        # 1. 缺失值处理
        nan_pct = initial_stats['missing_pct'] / 100.0
        factor_report["actions"].append(f"原始缺失值比例: {nan_pct:.2%}")
        if nan_pct > NAN_THRESHOLD_DISCARD:
            message = f"缺失值比例 {nan_pct:.2%} 高于阈值 {NAN_THRESHOLD_DISCARD:.2%}"
            factor_report["final_status"] = f"抛弃 ({message})"
            factor_report['status_value'] = -1
            if factor_name in df_corrected.columns:
                df_corrected.drop(columns=[factor_name], inplace=True)
            describe_factor_status(factor_name, "抛弃", message)
            report[factor_name] = factor_report
            continue
        elif nan_pct > 0:  # Impute if NaNs exist and are below threshold
            fill_value = np.nan
            # Use median/mean from initial_stats which are calculated on non-NaN data
            if NAN_IMPUTE_METHOD == 'median':
                fill_value = initial_stats['median']
            elif NAN_IMPUTE_METHOD == 'mean':
                fill_value = initial_stats['mean']

            if not np.isnan(fill_value):
                current_series.fillna(fill_value, inplace=True)
                action_msg = f"使用{NAN_IMPUTE_METHOD} ({fill_value:.4f}) 填充缺失值"
                factor_report["actions"].append(action_msg)
                print(action_msg)
            else:  # This case implies initial_stats['median/mean'] was NaN, meaning all original data was NaN or problematic
                # which should have been caught by initial_stats['count'] == 0. This is a safeguard.
                action_msg = f"无法填充缺失值 ({NAN_IMPUTE_METHOD} 计算为 NaN). 因子可能无效."
                factor_report["actions"].append(action_msg)
                print(action_msg)
                factor_report["final_status"] = "抛弃 (缺失值填充失败)"
                factor_report['status_value'] = -1
                if factor_name in df_corrected.columns:
                    df_corrected.drop(columns=[factor_name], inplace=True)
                describe_factor_status(factor_name, "抛弃", "缺失值填充失败")
                report[factor_name] = factor_report
                continue

        # Recalculate stats after potential imputation for subsequent checks
        stats_after_imputation = calculate_detailed_statistics(
            current_series, f"{factor_name} (填充后)")
        current_skew = stats_after_imputation['skewness']
        current_kurt = stats_after_imputation['kurtosis (excess)']
        current_std = stats_after_imputation['std_dev']
        current_mean = stats_after_imputation['mean']
        current_cv = stats_after_imputation['cv']
        current_ci_rel_width = stats_after_imputation['mean_ci_relative_width']
        current_count = stats_after_imputation['count']

        if current_count == 0:  # Should be caught by initial_stats, but as a safeguard after imputation
            message = "填充后无有效数据"
            factor_report["actions"].append(message)
            factor_report["final_status"] = f"抛弃 ({message})"
            factor_report['status_value'] = -1
            if factor_name in df_corrected.columns:
                df_corrected.drop(columns=[factor_name], inplace=True)
            describe_factor_status(factor_name, "抛弃", message)
            report[factor_name] = factor_report
            continue

        # 2. 标准差, CV, 置信区间等检查
        if np.isnan(current_std) or current_std < STD_DEV_THRESHOLD_DISCARD:
            message = f"标准差 ({current_std:.4g}) 过小或NaN, 因子近似常量或无效"
            factor_report["final_status"] = f"抛弃 ({message})"
            factor_report['status_value'] = -1
            if factor_name in df_corrected.columns:
                df_corrected.drop(columns=[factor_name], inplace=True)
            describe_factor_status(factor_name, "抛弃", message)
            report[factor_name] = factor_report
            continue
        # Add warnings
        if not np.isnan(current_cv) and abs(
                current_mean
        ) >= CV_VERY_LOW_MEAN_THRESHOLD and current_cv > CV_HIGH_THRESHOLD:
            warn_msg = f"高变异系数(CV): {current_cv:.2f}. 相对波动性大."
            factor_report["warnings"].append(warn_msg)
            print(f"警告: {warn_msg}")
        if not np.isnan(
                current_ci_rel_width
        ) and current_ci_rel_width > MEAN_CI_RELATIVE_WIDTH_THRESHOLD:
            warn_msg = f"均值置信区间相对宽度大: {current_ci_rel_width:.2%}. 均值估计可能不够精确."
            factor_report["warnings"].append(warn_msg)
            print(f"警告: {warn_msg}")

        # --- 开始修正流程 ---
        series_for_correction = current_series.copy(
        )  # This series is now NaN-filled
        transformation_log = []

        # Step A: Winsorization
        # current_skew/kurt are from *after imputation*
        if ENABLE_WINSORIZATION and (abs(current_skew) > SKEW_TARGET_MAX_ABS
                                     or abs(current_kurt)
                                     > KURTOSIS_TARGET_MAX_ABS):
            print(
                f"尝试Winsorization... (当前偏度:{current_skew:.2f}, 峰度:{current_kurt:.2f})"
            )
            non_nan_indices = series_for_correction.notna(
            )  # Should be all true if imputation worked
            if non_nan_indices.any():  # Still good practice to check
                winsorized_values = winsorize(
                    series_for_correction[non_nan_indices].values,
                    limits=WINSORIZE_LIMITS)
                series_for_correction.loc[non_nan_indices] = winsorized_values

                skew_after_winsor = skew(
                    series_for_correction.dropna())  # dropna for safety
                kurt_after_winsor = kurtosis(series_for_correction.dropna())
                action_msg = (
                    f"Winsorization ({WINSORIZE_LIMITS}) 应用. 偏度 {current_skew:.2f}->{skew_after_winsor:.2f}, 峰度 {current_kurt:.2f}->{kurt_after_winsor:.2f}"
                )
                transformation_log.append(action_msg)
                print(action_msg)
                current_skew = skew_after_winsor  # Update for next step
                current_kurt = kurt_after_winsor
                factor_report["transformation_applied"] = "Winsorization"

        # Step B: Shape Transformations
        # current_skew/kurt are now *after winsorization* (if applied)
        best_transformed_series = None
        best_transform_name_overall = factor_report[
            "transformation_applied"]  # Start with "Winsorization" or "无"
        best_transform_skew = current_skew
        best_transform_kurt = current_kurt
        best_transform_message = "无额外形状转换"
        best_applied_reflection_flag = False  # Track if the chosen transform involved reflection

        potential_transforms = []
        if ENABLE_LOG_TRANSFORM and abs(
                current_skew
        ) > LOG_TRANSFORM_MIN_SKEW_FOR_ATTEMPT:  # Check abs for general attempt
            potential_transforms.append('log1p')
        if ENABLE_SQRT_TRANSFORM and abs(
                current_skew) > SQRT_TRANSFORM_MIN_SKEW_FOR_ATTEMPT:
            potential_transforms.append('sqrt')
        if ENABLE_BOXCOX_TRANSFORM and abs(
                current_skew) > BOXCOX_MIN_ABS_SKEW_FOR_ATTEMPT:
            potential_transforms.append('boxcox')
        if ENABLE_YEOJOHNSON_TRANSFORM and abs(
                current_skew) > YEOJOHNSON_MIN_ABS_SKEW_FOR_ATTEMPT:
            potential_transforms.append('yeojohnson')

        # (Quick pre-checks for transform applicability can be done here, but apply_transformation handles them robustly)

        if potential_transforms:
            print(
                f"当前偏度 {current_skew:.2f} (Winsor后或填充后). 尝试形状转换: {potential_transforms}"
            )

        # `series_for_correction` is now the data after imputation and winsorization
        # `current_skew` and `current_kurt` are stats of this `series_for_correction`
        for transform_type in potential_transforms:
            # print(f"  尝试 {transform_type}...")
            transformed_s, message, lmbda, was_reflected = apply_transformation(
                series_for_correction,
                transform_type,
                current_skew,  # Pass skew of data being transformed
                current_kurt,  # Pass kurt of data being transformed
                ENABLE_REFLECTION_FOR_LEFT_SKEW)
            if transformed_s is not None:
                temp_skew_after_transform = skew(transformed_s.dropna())
                # temp_kurt_after_transform = kurtosis(transformed_s.dropna()) # This is in 'message'
                print(f" {transform_type} 结果: {message}")
                # Prefer transform that reduces skewness absolute value more
                if abs(temp_skew_after_transform) < abs(best_transform_skew):
                    best_transformed_series = transformed_s
                    current_best_transform_name_part = transform_type + (
                        "(R)" if was_reflected else "")
                    # Combine with previous step if any
                    if factor_report[
                            "transformation_applied"] != "无" and factor_report[
                                "transformation_applied"] != "Winsorization":
                        # This case should not happen if logic is: Winsor -> Shape
                        # Assuming transformation_applied is either "无" or "Winsorization" before this loop
                        best_transform_name_overall = f"{factor_report['transformation_applied']} + {current_best_transform_name_part}"
                    elif factor_report[
                            "transformation_applied"] == "Winsorization":
                        best_transform_name_overall = f"Winsorization + {current_best_transform_name_part}"
                    else:  # "无"
                        best_transform_name_overall = current_best_transform_name_part

                    best_transform_skew = temp_skew_after_transform
                    # best_transform_kurt can be extracted from message or recalculated
                    best_transform_message = message  # Store the full message
                    best_applied_reflection_flag = was_reflected
            # else:
            # print(f"    {transform_type} 失败/未改善: {message}")

        if best_transformed_series is not None:
            series_for_correction = best_transformed_series  # Update with the best one found
            # Get the last part of the transform name (e.g., log1p(R) from Winsorization + log1p(R))
            final_shape_transform_name_part = best_transform_name_overall.split(
                ' + ')[-1]
            transformation_log.append(
                f"应用 {final_shape_transform_name_part}: {best_transform_message}"
            )
            factor_report[
                "transformation_applied"] = best_transform_name_overall
            if best_applied_reflection_flag:
                reflection_note = f"注意: {final_shape_transform_name_part} 应用于反射后的数据。因子含义可能反转。"
                # This note is now part of best_transform_message if was_reflected, so covered by log.
                # factor_report["actions"].append(reflection_note) # Optional: explicit separate log
                print(reflection_note)
            print(
                f"最终选择的形状转换: {final_shape_transform_name_part} (偏度: {best_transform_skew:.2f})"
            )

        factor_report["actions"].extend(transformation_log)

        # Final stats after all transformations
        print("计算最终统计...")
        final_stats_obj = calculate_detailed_statistics(
            series_for_correction, f"{factor_name} (最终形态)")
        factor_report["final_stats_summary"] = {
            k: final_stats_obj[k]
            for k in [
                'mean', 'median', 'std_dev', 'cv', 'skewness',
                'kurtosis (excess)', 'missing_pct', 'count',
                'mean_ci_relative_width', 'skew_interpretation',
                'kurtosis_interpretation', 'mean_median_diff_norm_by_std'
            ]
        }
        final_skew = final_stats_obj['skewness']
        final_kurt = final_stats_obj['kurtosis (excess)']
        print(
            f"最终统计: 偏度={final_skew:.2f} ({final_stats_obj['skew_interpretation']}), 峰度={final_kurt:.2f} ({final_stats_obj['kurtosis_interpretation']})"
        )

        # 4. 最终决策
        passes_skew_test = abs(final_skew) <= SKEW_TARGET_MAX_ABS
        passes_kurt_test = abs(final_kurt) <= KURTOSIS_TARGET_MAX_ABS

        if passes_skew_test and passes_kurt_test:
            status_msg = "保留"
            if factor_report["transformation_applied"] != "无":
                status_msg += " (已修正)"
            else:
                status_msg += " (原始/填充后状态可接受)"
            reason_msg = f"最终偏度 {final_skew:.2f}, 最终超额峰度 {final_kurt:.2f} 均在目标范围内."
            factor_report["final_status"] = status_msg
            factor_report['status_value'] = 1
            describe_factor_status(factor_name, status_msg, reason_msg)
            df_corrected[factor_name] = series_for_correction
        else:
            reason_parts = []
            if not passes_skew_test:
                reason_parts.append(
                    f"最终偏度 ({final_skew:.2f}, {final_stats_obj['skew_interpretation']}) 超出目标 |S|<={SKEW_TARGET_MAX_ABS}"
                )
            if not passes_kurt_test:
                reason_parts.append(
                    f"最终超额峰度 ({final_kurt:.2f}, {final_stats_obj['kurtosis_interpretation']}) 超出目标 |EK|<={KURTOSIS_TARGET_MAX_ABS}"
                )
            full_reason = ". ".join(reason_parts)
            factor_report["final_status"] = f"抛弃 ({full_reason})"
            factor_report['status_value'] = -1
            if factor_name in df_corrected.columns:
                df_corrected.drop(columns=[factor_name], inplace=True)
            describe_factor_status(factor_name, "抛弃", full_reason)

            if abs(final_skew) > SKEW_HIGH_THRESHOLD or abs(
                    final_kurt) > KURTOSIS_HIGH_POSITIVE_THRESHOLD:
                biz_note = ("因子分布在修正后仍显著偏离正态。请结合业务逻辑判断此分布特性是否合理或为真实信号。")
                factor_report["warnings"].append(f"业务提示: {biz_note}")
                print(f"业务提示: {biz_note}")

        report[factor_name] = factor_report

    return df_corrected, report


# --- 辅助函数：滚动统计分析 (可选) ---
def analyze_factor_stability_rolling(data_series,
                                     factor_name,
                                     window=50,
                                     mean_median_std_threshold=0.3):
    """对单个因子进行滚动统计分析，评估其稳定性。"""
    if data_series.count() < window * 2:  # Need enough data for rolling window
        print(
            f"因子 {factor_name} 数据点不足 ({data_series.count()}) 以进行窗口为 {window} 的滚动分析。"
        )
        return

    print(f"\n{'='*10} 因子 '{factor_name}' 滚动统计分析 (窗口={window}) {'='*10}")

    rolling_mean = data_series.rolling(window=window).mean()
    rolling_median = data_series.rolling(window=window).median()
    rolling_std = data_series.rolling(window=window).std()

    # 1. 绘制滚动均值和中位数
    plt.figure(figsize=(12, 4))  # Adjusted figure size
    rolling_mean.plot(label='Rolling Mean', color='blue')
    rolling_median.plot(label='Rolling Median', color='orange')
    plt.title(f'滚动均值与滚动中位数: {factor_name} (窗口={window})')
    plt.xlabel('Time/Index')
    plt.ylabel('Factor Value')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()  # Ensure everything fits
    plt.show()
    print("观察上图：如果滚动均值和滚动中位数随时间剧烈波动或两者持续分离，可能表明特征不稳定或市场状态变化。")

    # 2. 计算 (mean - median) / std
    # Handle cases where rolling_std might be zero or very small or NaN
    rolling_mean_median_diff_norm = np.full_like(
        rolling_mean, np.nan)  # Initialize with NaNs

    # Mask for valid std (not NaN and not close to zero)
    valid_std_mask = rolling_std.notna() & (rolling_std.abs() > 1e-9)

    # Mask for valid mean and median (not NaN)
    valid_mean_median_mask = rolling_mean.notna() & rolling_median.notna()

    # Calculate for valid std and valid mean/median
    calc_mask = valid_std_mask & valid_mean_median_mask
    rolling_mean_median_diff_norm[calc_mask] = \
        (rolling_mean[calc_mask] - rolling_median[calc_mask]) / rolling_std[calc_mask]

    # Handle cases where std is zero (or very close to it)
    zero_std_mask = rolling_std.notna() & (rolling_std.abs() <= 1e-9)
    zero_std_calc_mask = zero_std_mask & valid_mean_median_mask  # Also need valid mean/median

    mean_eq_median_at_zero_std = zero_std_calc_mask & (rolling_mean
                                                       == rolling_median)
    mean_gt_median_at_zero_std = zero_std_calc_mask & (rolling_mean
                                                       > rolling_median)
    mean_lt_median_at_zero_std = zero_std_calc_mask & (rolling_mean
                                                       < rolling_median)

    rolling_mean_median_diff_norm[mean_eq_median_at_zero_std] = 0
    rolling_mean_median_diff_norm[mean_gt_median_at_zero_std] = np.inf
    rolling_mean_median_diff_norm[mean_lt_median_at_zero_std] = -np.inf

    rolling_mean_median_diff_norm_series = pd.Series(
        rolling_mean_median_diff_norm, index=data_series.index)
    plt.figure(figsize=(12, 4))  # Adjusted figure size
    rolling_mean_median_diff_norm_series.plot(
        label='(Rolling Mean - Rolling Median) / Rolling Std', color='green')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axhline(mean_median_std_threshold,
                color='red',
                linestyle=':',
                linewidth=0.8,
                label=f'Threshold ({mean_median_std_threshold})')
    plt.axhline(-mean_median_std_threshold,
                color='red',
                linestyle=':',
                linewidth=0.8)  # No label to avoid duplicate
    plt.title(f'滚动 (均值-中位数)/标准差: {factor_name} (窗口={window})')
    plt.xlabel('Time/Index')
    plt.ylabel('Normalized Difference')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()  # Ensure everything fits
    plt.show()
    print(
        f"观察上图：该值持续大于 {mean_median_std_threshold} 或小于 -{mean_median_std_threshold} 的区域，"
        "可能表明在那些时期特征存在显著的偏斜或受异常值影响。")


def select_base_factors(base_path,
                        method,
                        g_instruments,
                        types,
                        horizon,
                        n_top=-1):
    dirs = os.path.join(base_path, method, g_instruments, 'merged')
    ### 剔除空值的版本
    # o2o_1h_native_nofill.feather
    filename = os.path.join(
        dirs, "{0}_{1}h_{2}_{3}.feather".format(types, horizon, "native",
                                                'nofill'))
    total_data1 = pd.read_feather(filename)
    factor_cols = [
        col for col in total_data1.columns
        if col not in ['trade_time', 'code', 'nxt1_ret']
    ][:n_top]

    print("原始DataFrame因子描述 (部分列):")

    print(total_data1[factor_cols].describe(include='all'))

    print("\n开始筛选和修正因子...")
    df_corrected, report = screen_and_correct_factors(total_data1, factor_cols)
    #filename = os.path.join(dirs, "factors_base_report.feather")
    #factors_report.to_csv(filename)
    return df_corrected, factor_cols, report


if __name__ == "__main__":
    method = 'kimto1'
    g_instruments = 'rbb'
    horizon = 1
    category = 1
    types = 'o2o'
    ### 基础筛选+ 修复
    df_corrected, factor_cols, all_report = select_base_factors(
        base_path=base_path,
        method=method,
        g_instruments=g_instruments,
        types=types,
        horizon=horizon,
        n_top=-1)
    ### 计算IC时不是必须的，因为相关系数本身不受线性变换影响
    ### 收益率相关性筛选
    corr_res = []
    for col in factor_cols:
        if col in df_corrected.columns:
            corr1 = np.corrcoef(df_corrected[col], df_corrected['nxt1_ret'])[0,
                                                                             1]
            corr_res.append({'name': col, 'value': corr1})

    corr_pd = pd.DataFrame(corr_res)
    corr_pd['abs'] = corr_pd['value'].abs()
    corr_pd = corr_pd[(corr_pd['abs'] >= corr_pd['abs'].median()) & (corr_pd['abs'] >= 0.01)]
    factor_cols = corr_pd['name'].values.tolist()

    for name in all_report.keys():
        report = all_report[name]
        if name in factor_cols:
            report['status_value'] = 2
        report['name'] = name
        all_report[name] = report
    all_report = pd.DataFrame(all_report.values())
    dirs = os.path.join(base_path, method, g_instruments, 'merged')
    filename = os.path.join(dirs, "report.feather")
    all_report.to_feather(filename)