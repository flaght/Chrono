"""HybridTransformer 特征相关性与重复性审计。

审计只读取训练集，不填充缺失值，不对特征再次标准化，也不自动删除特征。
"""

from itertools import combinations
import json
import os
import time

import numpy as np
import pandas as pd


def _finite_numeric(frame, features):
    """转为数值并将正负无穷视为缺失；禁止填充为 0。"""
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def feature_quality(frame, features, code, old_features):
    values = _finite_numeric(frame, features)
    total = len(values)
    rows = []
    for feature in features:
        valid = values[feature].dropna()
        rows.append({
            "code": code,
            "feature_group": "old" if feature in old_features else "new",
            "feature": feature,
            "total_rows": total,
            "valid_rows": int(valid.size),
            "invalid_rows": int(total - valid.size),
            "valid_rate": float(valid.size / total) if total else np.nan,
            "mean": float(valid.mean()) if valid.size else np.nan,
            "std": float(valid.std()) if valid.size > 1 else np.nan,
            "nunique": int(valid.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def _pair_group(feature_1, feature_2, old_set):
    old_1 = feature_1 in old_set
    old_2 = feature_2 in old_set
    if old_1 and old_2:
        return "old_old"
    if not old_1 and not old_2:
        return "new_new"
    return "new_old"


def _pair_metrics(left, right, pearson_override=None, spearman_override=None):
    """在成对有效样本上计算相关性和数值差异。"""
    valid_mask = left.notna() & right.notna()
    overlap = int(valid_mask.sum())
    pearson = np.nan
    spearman = np.nan
    max_abs_diff = np.nan
    mean_abs_diff = np.nan
    overlap_equal = False
    if overlap:
        left_valid = left[valid_mask]
        right_valid = right[valid_mask]
        diff = (left_valid - right_valid).abs()
        max_abs_diff = float(diff.max())
        mean_abs_diff = float(diff.mean())
        overlap_equal = bool(np.array_equal(left_valid.to_numpy(),
                                            right_valid.to_numpy(),
                                            equal_nan=True))
        if (overlap >= 2 and left_valid.nunique() > 1
                and right_valid.nunique() > 1):
            pearson = (float(pearson_override)
                       if pearson_override is not None else
                       float(left_valid.corr(right_valid, method="pearson")))
            # Spearman 等于秩变量的 Pearson；这样不额外依赖 scipy。
            spearman = (float(spearman_override)
                        if spearman_override is not None else
                        float(left_valid.rank(method="average").corr(
                            right_valid.rank(method="average"),
                            method="pearson")))
    same_missing_mask = bool(left.isna().equals(right.isna()))
    return {
        "overlap": overlap,
        "pearson": pearson,
        "spearman": spearman,
        "same_missing_mask": same_missing_mask,
        "exact_equal": bool(same_missing_mask and overlap_equal),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def correlate_all_pairs(frame, features, old_features, code,
                        progress_every=50):
    """计算单品种全部特征组合，包括旧旧、新旧和新新。"""
    values = _finite_numeric(frame, features)
    old_set = set(old_features)
    pairs = list(combinations(features, 2))
    # 常见训练文件已经没有缺失值。此时每列仅排名一次，避免对每一对
    # 特征重复执行 O(n log n) 的 Spearman 排名。
    complete = not bool(values.isna().to_numpy().any())
    pearson_matrix = None
    spearman_matrix = None
    if complete:
        matrix_started = time.time()
        pearson_matrix = values.corr(method="pearson")
        spearman_matrix = values.rank(method="average").corr(method="pearson")
        print(f"[FEATURE_CORR_MATRIX] code={code} rows={len(values)} "
              f"features={len(features)} elapsed="
              f"{time.time() - matrix_started:.1f}s")
    rows = []
    started = time.time()
    for pair_no, (feature_1, feature_2) in enumerate(pairs, 1):
        pearson_override = (pearson_matrix.at[feature_1, feature_2]
                            if complete else None)
        spearman_override = (spearman_matrix.at[feature_1, feature_2]
                             if complete else None)
        metrics = _pair_metrics(values[feature_1], values[feature_2],
                                pearson_override, spearman_override)
        row = {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "pair_group": _pair_group(feature_1, feature_2, old_set),
        }
        row.update({f"{code}_{key}": value for key, value in metrics.items()})
        rows.append(row)
        if progress_every and (pair_no % progress_every == 0
                               or pair_no == len(pairs)):
            elapsed = time.time() - started
            print(f"[FEATURE_CORR_PROGRESS] code={code} "
                  f"pairs={pair_no}/{len(pairs)} elapsed={elapsed:.1f}s")
    return pd.DataFrame(rows)


def _asset_is_strong(table, code, threshold):
    return ((table[f"{code}_pearson"].abs() >= threshold)
            & (table[f"{code}_spearman"].abs() >= threshold))


def classify_pairs(table, codes=("RB", "HC"), strong_threshold=0.85,
                   duplicate_threshold=0.95):
    """按照双品种共同成立原则分类，不执行自动删除。"""
    result = table.copy()
    duplicate_flags = [
        _asset_is_strong(result, code, duplicate_threshold) for code in codes
    ]
    strong_flags = [
        _asset_is_strong(result, code, strong_threshold) for code in codes
    ]
    exact_flags = [result[f"{code}_exact_equal"].fillna(False) for code in codes]
    result["duplicate_candidate"] = np.logical_and.reduce(duplicate_flags)
    result["exact_duplicate_all_assets"] = np.logical_and.reduce(exact_flags)
    result["strong_both_assets"] = np.logical_and.reduce(strong_flags)
    result["asset_specific_strong"] = (
        np.column_stack(strong_flags).sum(axis=1) == 1)
    result["classification"] = "RETAIN"
    result.loc[result["asset_specific_strong"],
               "classification"] = "ASSET_SPECIFIC_OBSERVATION"
    result.loc[result["strong_both_assets"],
               "classification"] = "STRONG_OBSERVATION"
    result.loc[result["duplicate_candidate"],
               "classification"] = "EXTREME_DUPLICATE_CANDIDATE"
    result.loc[result["exact_duplicate_all_assets"],
               "classification"] = "EXACT_DUPLICATE"
    correlation_columns = [
        f"{code}_{kind}" for code in codes for kind in ("pearson", "spearman")
    ]
    result["min_abs_corr_all"] = result[correlation_columns].abs().min(axis=1)
    result["max_abs_corr_all"] = result[correlation_columns].abs().max(axis=1)
    return result.sort_values(
        ["exact_duplicate_all_assets", "duplicate_candidate",
         "min_abs_corr_all"], ascending=[False, False, False]
    ).reset_index(drop=True)


def _legacy_new_old_table(result, old_features):
    """保留原新×旧输出字段，方便已有 notebook 继续读取。"""
    old_set = set(old_features)
    legacy = result[result["pair_group"] == "new_old"].copy()
    legacy["new_feature"] = np.where(
        legacy["feature_1"].isin(old_set), legacy["feature_2"],
        legacy["feature_1"])
    legacy["old_feature"] = np.where(
        legacy["feature_1"].isin(old_set), legacy["feature_1"],
        legacy["feature_2"])
    first = ["new_feature", "old_feature"]
    dropped = {"feature_1", "feature_2", "pair_group"}
    remaining = [column for column in legacy.columns
                 if column not in first and column not in dropped]
    return legacy[first + remaining].reset_index(drop=True)


def feature_correlation(asset_frames, old_features, new_features, output_dir,
                        strong_threshold=0.85,
                        duplicate_threshold=0.95):
    """执行完整 40×40 审计并保存全量及兼容版结果。"""
    if set(old_features) & set(new_features):
        raise ValueError("old_features 与 new_features 不能重叠")
    if not old_features or not new_features:
        raise ValueError("新旧特征列表均不能为空")
    if len(asset_frames) < 2:
        raise ValueError("至少需要两个品种，当前规则要求 RB、HC 交叉确认")
    os.makedirs(output_dir, exist_ok=True)
    all_features = list(old_features) + list(new_features)
    if len(set(all_features)) != len(all_features):
        raise ValueError("特征列表存在重复名称")

    correlations = []
    qualities = []
    for code, frame in asset_frames.items():
        missing = set(all_features) - set(frame.columns)
        if missing:
            raise ValueError(f"{code} 训练集缺少特征: {sorted(missing)}")
        correlations.append(
            correlate_all_pairs(frame, all_features, old_features, code))
        qualities.append(
            feature_quality(frame, all_features, code, set(old_features)))

    join_columns = ["feature_1", "feature_2", "pair_group"]
    merged = correlations[0]
    for other in correlations[1:]:
        merged = merged.merge(other, on=join_columns, how="outer",
                              validate="one_to_one")
    codes = tuple(asset_frames)
    result = classify_pairs(merged, codes, strong_threshold,
                            duplicate_threshold)
    legacy = _legacy_new_old_table(result, old_features)
    quality = pd.concat(qualities, ignore_index=True)
    observations = result[result["classification"] != "RETAIN"].copy()
    candidates = result[result["duplicate_candidate"]].copy()
    exact_duplicates = result[result["exact_duplicate_all_assets"]].copy()

    result.to_csv(os.path.join(output_dir, "feature_all_pairs.csv"), index=False)
    exact_duplicates.to_csv(os.path.join(output_dir,
                                         "feature_exact_duplicates.csv"),
                            index=False)
    observations.to_csv(os.path.join(output_dir,
                                     "feature_all_observations.csv"),
                        index=False)
    candidates.to_csv(os.path.join(
        output_dir, "feature_all_duplicate_candidates.csv"), index=False)
    legacy.to_csv(os.path.join(output_dir, "feature_correlation_pairs.csv"),
                  index=False)
    legacy[legacy["classification"] != "RETAIN"].to_csv(
        os.path.join(output_dir, "feature_correlation_observations.csv"),
        index=False)
    legacy[legacy["duplicate_candidate"]].to_csv(
        os.path.join(output_dir, "feature_duplicate_candidates.csv"),
        index=False)
    quality.to_csv(os.path.join(output_dir, "feature_data_quality.csv"),
                   index=False)

    group_counts = result.groupby("pair_group").size().to_dict()
    exact_group_counts = exact_duplicates.groupby("pair_group").size().to_dict()
    summary = {
        "scope": "train_only_all_feature_pairs",
        "codes": list(codes),
        "old_feature_count": len(old_features),
        "new_feature_count": len(new_features),
        "feature_count": len(all_features),
        "all_pair_count": int(len(result)),
        "expected_all_pair_count": int(len(all_features)
                                       * (len(all_features) - 1) / 2),
        "pair_group_counts": {key: int(value)
                              for key, value in group_counts.items()},
        "strong_threshold": float(strong_threshold),
        "duplicate_threshold": float(duplicate_threshold),
        "duplicate_candidate_count": int(result["duplicate_candidate"].sum()),
        "exact_duplicate_count": int(
            result["exact_duplicate_all_assets"].sum()),
        "exact_duplicate_group_counts": {
            key: int(value) for key, value in exact_group_counts.items()
        },
        "strong_both_assets_count": int(result["strong_both_assets"].sum()),
        "asset_specific_strong_count": int(
            result["asset_specific_strong"].sum()),
        "old_features": list(old_features),
        "new_features": list(new_features),
        "rules": {
            "missing_values": "pairwise_drop; never fill with zero",
            "standardization": "none",
            "exact_duplicate": (
                "identical missing mask and exactly equal finite values "
                "for every asset"),
            "duplicate_candidate": (
                "abs(Pearson) and abs(Spearman) >= duplicate_threshold "
                "for every asset"),
            "automatic_deletion": False,
        },
    }
    with open(os.path.join(output_dir, "feature_correlation_summary.json"),
              "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"[FEATURE_CORR_END] output={output_dir} pairs={len(result)} "
          f"exact_duplicates={len(exact_duplicates)} "
          f"duplicate_candidates={len(candidates)}")
    return result, quality, summary
