import pandas as pd


def judge_price_field(row):
    name = row["name"]

    if name in ["open", "close"]:
        if (row["within_1tick_ratio"] >= 0.995
                and row["p99_abs_diff_tick"] <= 1
                and row["max_abs_diff_tick"] <= 3):
            return "PASS"

        if (row["within_1tick_ratio"] >= 0.95
                and row["within_2tick_ratio"] >= 0.99
                and row["p99_abs_diff_tick"] <= 2
                and row["max_abs_diff_tick"] <= 5):
            return "WARN"

        return "FAIL"

    if name in ["high", "low"]:
        if (row["within_1tick_ratio"] >= 0.999
                and row["p99_abs_diff_tick"] <= 1
                and row["max_abs_diff_tick"] <= 2):
            return "PASS"

        if (row["within_1tick_ratio"] >= 0.98
                and row["within_2tick_ratio"] >= 0.995
                and row["p99_abs_diff_tick"] <= 2
                and row["max_abs_diff_tick"] <= 5):
            return "WARN"

        return "FAIL"

    if name == "vwap":
        if (row["within_1tick_ratio"] >= 0.999
                and row["p99_abs_diff_tick"] <= 0.5
                and row["max_abs_diff_tick"] <= 1):
            return "PASS"

        if (row["within_1tick_ratio"] >= 0.99 and row["p99_abs_diff_tick"] <= 1
                and row["max_abs_diff_tick"] <= 2):
            return "WARN"

        return "FAIL"

    return "CHECK"


def judge_rel_field(row):
    name = row["name"]

    if name in ["volume", "value"]:
        if (row["median_rel_diff"] <= 0.001 and row["p95_rel_diff"] <= 0.02
                and row["p99_rel_diff"] <= 0.05
                and row["large_diff_5pct_ratio"] <= 0.01):
            return "PASS"

        if (row["median_rel_diff"] <= 0.005 and row["p95_rel_diff"] <= 0.10
                and row["p99_rel_diff"] <= 0.25
                and row["large_diff_5pct_ratio"] <= 0.10):
            return "WARN"

        return "FAIL"

    if name == "openint":
        if (row["p95_rel_diff"] <= 0.0001 and row["p99_rel_diff"] <= 0.0005
                and row["max_rel_diff"] <= 0.001):
            return "PASS"

        if (row["p95_rel_diff"] <= 0.001 and row["p99_rel_diff"] <= 0.005
                and row["max_rel_diff"] <= 0.01):
            return "WARN"

        return "FAIL"

    return "CHECK"


def judge_overall_bar_status(field_status):
    price_fields = ["open", "high", "low", "close", "vwap"]
    volume_fields = ["volume", "value"]

    statuses = list(field_status.values())

    if "FAIL" in statuses:
        return "FAIL"

    if "CHECK" in statuses:
        return "CHECK"

    if all(s == "PASS" for s in statuses):
        return "PASS"

    price_all_pass = all(field_status.get(x) == "PASS" for x in price_fields)
    volume_has_warn = any(field_status.get(x) == "WARN" for x in volume_fields)
    only_volume_warn = all(
        field_status.get(x) == "PASS"
        for x in ["open", "high", "low", "close", "vwap", "openint"
                  ]) and volume_has_warn

    if only_volume_warn:
        return "PASS_WARN_VOLUME"

    return "WARN"


def generate_bar_status(price_metrics: pd.DataFrame,
                        rel_metrics: pd.DataFrame):
    price_metrics = price_metrics.copy()
    rel_metrics = rel_metrics.copy()

    price_metrics["status"] = price_metrics.apply(judge_price_field, axis=1)
    rel_metrics["status"] = rel_metrics.apply(judge_rel_field, axis=1)

    field_status = {}

    for _, row in price_metrics.iterrows():
        field_status[row["name"]] = row["status"]

    for _, row in rel_metrics.iterrows():
        field_status[row["name"]] = row["status"]

    overall_status = judge_overall_bar_status(field_status)

    return {
        "overall_status": overall_status,
        "field_status": field_status,
        "price_metrics": price_metrics,
        "rel_metrics": rel_metrics,
    }
