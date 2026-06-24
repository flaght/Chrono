from lib.attr001.ftd001 import *
from lib.attr001.ftd002 import *




def judge_overall(field_status, price_fields, flow_fields) -> str:
    statuses = list(field_status.values())

    if "FAIL" in statuses:
        return "FAIL"
    if "CHECK" in statuses:
        return "CHECK"
    if all(status == "PASS" for status in statuses):
        return "PASS"

    price_all_pass = all(
        field_status.get(field) == "PASS" for field in price_fields)
    only_volume_warn = (price_all_pass
                        and field_status.get("openint") == "PASS" and any(
                            field_status.get(field) == "WARN"
                            for field in flow_fields) and all(
                                field_status.get(field) in ["PASS", "WARN"]
                                for field in flow_fields))

    if only_volume_warn:
        return "PASS_WARN_VOLUME"
    return "WARN"


def judge_flow_field(row) -> str:
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


def judge_price_field(row) -> str:
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


def diff_detail(research_data,
                trader_data,
                tick_size,
                adjusted_method,
                price_columns=['open', 'close', 'high', 'low', 'vwap'],
                flow_columns=['volume', 'openint']):

    research_data, trader_data = algin_data2(research_data, trader_data)
    merge_data = pd.DataFrame(index=trader_data.index)
    if adjusted_method is None:
        merge_data['effective_tick_size'] = tick_size
    else:
        factor_col = "{0}_cumfactor".format(adjusted_method)
        if factor_col not in research_data.columns:
            raise ValueError(
                "missing adjust factor column: {0}".format(factor_col))
        if research_data[factor_col].isna().any():
            raise ValueError(
                "adjust factor column contains NaN: {0}".format(factor_col))
        if (research_data[factor_col] <= 0).any():
            raise ValueError(
                "adjust factor column contains non-positive values: {0}".
                format(factor_col))
        merge_data[
            'effective_tick_size'] = tick_size * research_data[factor_col]

    for field in price_columns:
        diff = research_data[field] - trader_data[field]
        merge_data[f"{field}_diff"] = diff
        merge_data[f"{field}_abs_diff_tick"] = diff.abs() / merge_data[
            'effective_tick_size']  # 从价格差转化为相差多少个最小单位变动。 期货

    for field in flow_columns:
        diff = research_data[field] - trader_data[field]
        base = pd.concat(
            [research_data[field].abs(), trader_data[field].abs()],
            axis=1,
        ).max(axis=1)
        rel_diff = diff.abs() / base.replace(0, np.nan)
        rel_diff = rel_diff.fillna(0.0)
        merge_data[f"{field}_diff"] = diff
        merge_data[f"{field}_abs_diff"] = diff.abs()
        merge_data[f"{field}_rel_diff"] = rel_diff

    return merge_data


def price_metrics(details, field):
    diff_tick = details[f"{field}_abs_diff_tick"]
    valid_count = int(diff_tick.notna().sum())
    if valid_count == 0:
        return {
            "name": field,
            "field_type": "price",
            "valid_count": 0,
            "exact_match_ratio": float("nan"),
            "within_1tick_ratio": float("nan"),
            "within_2tick_ratio": float("nan"),
            "mean_abs_diff_tick": float("nan"),
            "median_abs_diff_tick": float("nan"),
            "p95_abs_diff_tick": float("nan"),
            "p99_abs_diff_tick": float("nan"),
            "max_abs_diff_tick": float("nan"),
        }

    return {
        "name": field,
        "field_type": "price",
        "valid_count": valid_count,
        "exact_match_ratio":
        float(np.isclose(diff_tick, 0, atol=FLOAT_EPS).mean()),
        "within_1tick_ratio": float(diff_tick.le(1 + FLOAT_EPS).mean()),
        "within_2tick_ratio": float(diff_tick.le(2 + FLOAT_EPS).mean()),
        "mean_abs_diff_tick": safe_mean(diff_tick),
        "median_abs_diff_tick": safe_median(diff_tick),
        "p95_abs_diff_tick": safe_quantile(diff_tick, 0.95),
        "p99_abs_diff_tick": safe_quantile(diff_tick, 0.99),
        "max_abs_diff_tick": float(diff_tick.max()),
    }


def flow_metrics(detail: pd.DataFrame, field: str):
    abs_diff = detail[f"{field}_abs_diff"]
    rel_diff = detail[f"{field}_rel_diff"]
    valid_count = int(rel_diff.notna().sum())
    if valid_count == 0:
        return {
            "name": field,
            "field_type": "relative",
            "valid_count": 0,
            "exact_match_ratio": float("nan"),
            "mean_abs_diff": float("nan"),
            "median_abs_diff": float("nan"),
            "p95_abs_diff": float("nan"),
            "p99_abs_diff": float("nan"),
            "max_abs_diff": float("nan"),
            "mean_rel_diff": float("nan"),
            "median_rel_diff": float("nan"),
            "p95_rel_diff": float("nan"),
            "p99_rel_diff": float("nan"),
            "max_rel_diff": float("nan"),
            "large_diff_1pct_ratio": float("nan"),
            "large_diff_2pct_ratio": float("nan"),
            "large_diff_5pct_ratio": float("nan"),
        }

    return {
        "name": field,
        "field_type": "relative",
        "valid_count": valid_count,
        "exact_match_ratio": float(abs_diff.eq(0).mean()),
        "mean_abs_diff": safe_mean(abs_diff),
        "median_abs_diff": safe_median(abs_diff),
        "p95_abs_diff": safe_quantile(abs_diff, 0.95),
        "p99_abs_diff": safe_quantile(abs_diff, 0.99),
        "max_abs_diff": float(abs_diff.max()),
        "mean_rel_diff": safe_mean(rel_diff),
        "median_rel_diff": safe_median(rel_diff),
        "p95_rel_diff": safe_quantile(rel_diff, 0.95),
        "p99_rel_diff": safe_quantile(rel_diff, 0.99),
        "max_rel_diff": float(rel_diff.max()),
        "large_diff_1pct_ratio": float(rel_diff.gt(0.01).mean()),
        "large_diff_2pct_ratio": float(rel_diff.gt(0.02).mean()),
        "large_diff_5pct_ratio": float(rel_diff.gt(0.05).mean()),
    }


def metrics_price(details, price_fields):
    res = []
    for col in price_fields:
        if f"{col}_abs_diff_tick" not in details.columns:
            continue
        res.append(price_metrics(details, col))
    return res


def metrics_flow(details, flow_fields):
    res = []
    for col in flow_fields:
        if f"{col}_abs_diff" not in details.columns:
            continue
        res.append(flow_metrics(details, col))
    return res


def find_bar_anomalies(
    details,
    close_diff_tick_threshold: float = 1.0,
    volume_rel_diff_threshold: float = 0.05,
) -> pd.DataFrame:
    anomaly_mask = pd.Series(False, index=details.index)
    if "close_abs_diff_tick" in details.columns:
        anomaly_mask |= details[
            "close_abs_diff_tick"] > close_diff_tick_threshold + FLOAT_EPS
    if "volume_rel_diff" in details.columns:
        anomaly_mask |= details["volume_rel_diff"] > volume_rel_diff_threshold
    if "openint_rel_diff" in details.columns:
        anomaly_mask |= details["openint_rel_diff"] > volume_rel_diff_threshold

    anomalies = details.loc[anomaly_mask].copy()
    if anomalies.empty:
        return anomalies

    if "close_abs_diff_tick" in anomalies.columns:
        anomalies["close_diff_tick"] = anomalies["close_abs_diff_tick"]
    if "open_abs_diff_tick" in anomalies.columns:
        anomalies["open_diff_tick"] = anomalies["open_abs_diff_tick"]
    if "high_abs_diff_tick" in anomalies.columns:
        anomalies["high_diff_tick"] = anomalies["high_abs_diff_tick"]
    if "low_abs_diff_tick" in anomalies.columns:
        anomalies["low_diff_tick"] = anomalies["low_abs_diff_tick"]
    if "vwap_abs_diff_tick" in anomalies.columns:
        anomalies["vwap_diff_tick"] = anomalies["vwap_abs_diff_tick"]

    anomalies = anomalies.sort_values(["trade_time",
                                       "code"]).reset_index(drop=True)
    return anomalies


def diagnostics(research_data,
                trader_data,
                tick_size,
                adjusted_method,
                price_columns=['open', 'close', 'high', 'low', 'vwap'],
                flow_columns=['volume', 'openint']):

    rows_price = []
    rows_flow = []
    details = diff_detail(research_data=research_data,
                          trader_data=trader_data,
                          tick_size=tick_size,
                          adjusted_method=adjusted_method,
                          price_columns=price_columns,
                          flow_columns=flow_columns)

    for row in metrics_price(details=details, price_fields=price_columns):
        row["status"] = judge_price_field(row)
        rows_price.append(row)

    for row in metrics_flow(details=details, flow_fields=flow_columns):
        row["status"] = judge_flow_field(row)
        rows_flow.append(row)

    
    metrics_price1 = pd.DataFrame(rows_price)
    metrics_flow1 = pd.DataFrame(rows_flow)
    field_status = dict(
        zip(metrics_price1["name"],
            metrics_price1["status"])) if not metrics_price1.empty else {}

    field_status.update(
        dict(zip(metrics_flow1["name"], metrics_flow1["status"])
             ) if not metrics_flow1.empty else {})

    overall_status = judge_overall(
        field_status=field_status,
        price_fields=price_columns,
        flow_fields=flow_columns) if field_status else "CHECK"

    return {
        "summary": {
            "aligned_bar_count": len(details),
            "field_status": field_status,
            "bar_status": overall_status,
        },
        "metrics": {
            "price": metrics_price1,
            "flow": metrics_flow1
        },
        "details": details,
        "anomalies": find_bar_anomalies(details=details),
    }
