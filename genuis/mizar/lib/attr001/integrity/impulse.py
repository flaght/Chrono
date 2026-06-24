import itertools
from ultron.sentry.api import *
from lumina.formual.impulse import Impulse
from lib.attr001.logic001 import *
from lib.attr001.ftd002 import *


def run_factors1(factors_infos, market_data):
    market_res = market_data_format(market_data)
    dependencies = [
        eval(formula['formula'])._dependency for formula in factors_infos
    ]
    dependencies = list(itertools.chain.from_iterable(dependencies))
    factors_data1 = Impulse(dependencies).batch(data=market_res)
    return factors_data1


def feature_metrics(research_data, trader_data, feature_columns):
    rows = []
    for feature in feature_columns:
        diff = research_data[feature] - trader_data[feature]
        abs_diff = diff.abs()
        denom = pd.concat(
            [research_data[feature].abs(), trader_data[feature].abs()],
            axis=1).max(axis=1)
        rel_diff = abs_diff / denom
        zero_cross = (research_data[feature] * trader_data[feature]) < 0
        row = {
            "name":
            feature,
            "valid_count":
            int(len(research_data[feature])),
            "mean_abs_diff":
            safe_mean(abs_diff),
            "median_abs_diff":
            safe_median(abs_diff),
            "p95_abs_diff":
            safe_quantile(abs_diff, 0.95),
            "p99_abs_diff":
            safe_quantile(abs_diff, 0.99),
            "max_abs_diff":
            float(abs_diff.max()),
            "mean_rel_diff":
            safe_mean(rel_diff),
            "median_rel_diff":
            safe_median(rel_diff),
            "p95_rel_diff":
            safe_quantile(rel_diff, 0.95),
            "p99_rel_diff":
            safe_quantile(rel_diff, 0.99),
            "max_rel_diff":
            float(rel_diff.max()),
            "pearson_corr":
            safe_corr(research_data[feature], trader_data[feature], "pearson"),
            "spearman_corr":
            safe_corr(research_data[feature], trader_data[feature],
                      "spearman"),
            "sign_match_ratio":
            float((np.sign(research_data[feature]) == np.sign(
                trader_data[feature])).mean()),
            "zero_cross_ratio":
            float(zero_cross.mean()),
        }
        rows.append(row)
    return rows


def state_diff(research_data, trader_data):
    diff = research_data - trader_data
    out = pd.DataFrame(index=research_data.index)
    out["state_l1"] = diff.abs().sum(axis=1)
    out["state_l2"] = np.sqrt((diff**2).sum(axis=1))
    out["state_max_abs_diff"] = diff.abs().max(axis=1)
    out["state_nan_count"] = diff.isna().sum(axis=1)

    dot = (research_data * trader_data).sum(axis=1)
    nr = np.sqrt((research_data**2).sum(axis=1))
    nt = np.sqrt((trader_data**2).sum(axis=1))
    denom = (nr * nt).replace(0, np.nan)
    out["state_cosine"] = (dot / denom).fillna(0.0)

    median_l2 = out["state_l2"].median()
    p99_l2 = out["state_l2"].quantile(0.99)
    threshold = max(median_l2 * 3, p99_l2)
    out["state_outlier"] = out["state_l2"] > threshold

    return out


def state_metrics(state_detail):
    return {
        "valid_count":
        int(len(state_detail)),
        "state_l1_median":
        safe_median(state_detail["state_l1"]
                    ),  # 每个时点，所有因子差值绝对值求和；再看中位数。很接近 0，说明大部分时点总偏差极小。
        "state_l2_median":
        safe_median(state_detail["state_l2"]
                    ),  # 每个时点，所有因子差值做欧氏距离；再看中位数。也接近 0，说明整体 state 向量几乎重合。
        "state_l2_p95":
        safe_quantile(state_detail["state_l2"],
                      0.95),  # 说明大多数时点差异很小，但尾部有少数较大偏差点。
        "state_l2_p99":
        safe_quantile(state_detail["state_l2"], 0.99),
        "state_max_abs_diff_p95":
        safe_quantile(state_detail["state_max_abs_diff"],
                      0.95),  # 每个时点里，单个因子的最大偏差的 95 分位。用来判断“最差单因子偏差”通常有多大。
        "state_cosine_median":
        safe_median(
            state_detail["state_cosine"]),  # state 向量方向一致性。1.0 几乎是完美一致。
        "state_cosine_p05":
        safe_quantile(state_detail["state_cosine"],
                      0.05),  # 即使在较差的 5% 样本里，方向也几乎完全一致。
        "state_nan_count_sum":
        int(state_detail["state_nan_count"].sum()
            ),  # 总共出现了多少 NaN 差异位点，8 不算多，但值得知道。
        "state_outlier_count":
        int(state_detail["state_outlier"].sum()
            ),  # 有多少时点被判成异常点。这里约 1.1%，说明不是全局性问题，而是局部问题。
        "state_outlier_ratio":
        float(state_detail["state_outlier"].mean()),
    }


def top_feature_diff(research_data, trader_data, top_n=10):
    diff = (research_data - trader_data).abs()
    contribution = diff.mean(axis=0).sort_values(ascending=False)
    out = contribution.head(top_n).reset_index()
    out.columns = ["feature_name", "mean_abs_diff"]
    return out


def judge_state_status(state_metrics):
    if state_metrics["state_cosine_median"] >= 0.995:
        return "PASS"
    if state_metrics["state_cosine_median"] >= 0.98:
        return "WARN"
    return "FAIL"


def build_state_anomalies(state_detail, top_feature_contrib, limit=100):
    if state_detail.empty:
        return state_detail

    anomalies = state_detail.loc[state_detail["state_outlier"]].copy()
    if anomalies.empty:
        anomalies = state_detail.sort_values(
            ["state_l2", "state_max_abs_diff"],
            ascending=False).head(limit).copy()
    else:
        anomalies = anomalies.sort_values(["state_l2", "state_max_abs_diff"],
                                          ascending=False).head(limit).copy()

    if not top_feature_contrib.empty:
        anomalies["top_feature_hint"] = ",".join(
            top_feature_contrib["feature_name"].head(5).tolist())
    return anomalies


def diagnostics(factors_infos, research_data, trader_data, top_n=10):
    ## 对齐数据
    research_factors = run_factors1(factors_infos=factors_infos,
                                    market_data=research_data)
    trader_factors = run_factors1(factors_infos=factors_infos,
                                  market_data=trader_data)
    return diagnostics1(research_factors=research_factors,
                        trader_factors=trader_factors,
                        top_n=top_n)


def diagnostics1(research_factors, trader_factors, top_n=10):
    research_factors, trader_factors = algin_data2(research_factors,
                                                   trader_factors)

    feature_columns = list(
        set(research_factors.columns) & set(trader_factors.columns))

    research_factors = research_factors[feature_columns].sort_index()
    trader_factors = trader_factors[feature_columns].sort_index()

    state_detail = state_diff(research_data=research_factors,
                              trader_data=trader_factors)

    state_metrics1 = state_metrics(state_detail)

    per_feature_metrics = feature_metrics(research_data=research_factors,
                                          trader_data=trader_factors,
                                          feature_columns=feature_columns)

    top_features = top_feature_diff(research_data=research_factors,
                                    trader_data=trader_factors,
                                    top_n=top_n)

    state_status = judge_state_status(state_metrics1)

    anomalies = build_state_anomalies(state_detail, top_features)

    return {
        "summary": {
            **state_metrics1,
            "state_status": state_status,
            "feature_count": int(len(feature_columns)),
        },
        "state_detail": state_detail,
        "feature_metrics": per_feature_metrics,
        "top_feature_diff": top_features,
        "anomalies": anomalies,
        "research_factors": research_factors,
        "trader_factors": trader_factors,
    }
