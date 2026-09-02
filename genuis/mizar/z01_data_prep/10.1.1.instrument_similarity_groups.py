from dotenv import load_dotenv

load_dotenv()

from kdutils.data import *

EPS = 1e-12

# 数据是否足够 # 表示两个品种至少需要有504个共同有效交易日，才能计算它们之间的统计距离。
MIN_OVERLAP_DAYS = 504
# 候选与当前组整体是否足够接近 # 表示候选品种与当前组所有成员的平均距离，最多允许达到A组核心距离的1.25倍
MAX_AVG_RATIO = 1.35
# 候选是否与某个成员严重不兼容； # 表示候选与当前组中最远成员的距离，最多允许达到A组核心距离的1.35倍。
MAX_PAIR_RATIO = 1.50
# 这种相似关系是否在历史上长期稳定 # 表示候选品种必须在至少70%的滚动历史窗口中通过扩组条件，才能成为固定组成员。
MIN_INCLUSION_RATE = 0.70

WINDOW_DAYS = 756
STEP_DAYS = 252

CORR_WEIGHT = 0.5
FINGERPRINT_WEIGHT = 0.5


def statistical_fingerprint(ret: pd.Series) -> np.ndarray:
    """
    使用日收益构建简化指纹。

    所有分量尽量无量纲化，避免不同品种价格尺度影响：
    偏度、超额峰度、涨跌比例、尾部比、收益 ACF1/5、绝对收益 ACF1/5。
    """
    x = pd.to_numeric(ret, errors="coerce").dropna()
    if len(x) < 30:
        return np.full(8, np.nan)
    sigma = float(x.std(ddof=1))
    if not np.isfinite(sigma) or sigma < EPS:
        return np.full(8, np.nan)
    z = x / sigma
    q05, q95 = np.quantile(z, [0.05, 0.95])

    return np.array(
        [
            float(z.skew()),
            float(z.kurt()),
            float((z > 0).mean()),
            float(abs(q05) / (abs(q95) + EPS)),
            float(z.autocorr(1)),
            float(z.autocorr(5)),
            float(z.abs().autocorr(1)),
            float(z.abs().autocorr(5)),
        ],
        dtype=float,
    )


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x1, y1 = x[mask], y[mask]
    if np.std(x1) < EPS or np.std(y1) < EPS:
        return np.nan
    return float(np.corrcoef(x1, y1)[0, 1])


### 统一量纲
def _robust_standardize_rows(values: np.ndarray) -> np.ndarray:
    """按指纹维度在品种间 robust scaling。"""
    med = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > EPS, scale, fallback)
    scale = np.where(scale > EPS, scale, 1.0)
    return (values - med) / scale


def build_pairwise_distance(returns: pd.DataFrame):
    symbols = list(returns.columns)
    fingerprints = np.vstack(
        [statistical_fingerprint(returns[s]) for s in symbols])
    fp_scaled = _robust_standardize_rows(fingerprints)

    distance = pd.DataFrame(np.nan,
                            index=symbols,
                            columns=symbols,
                            dtype=float)
    overlap = pd.DataFrame(0, index=symbols, columns=symbols, dtype=int)

    for i, left in enumerate(symbols):
        for j, right in enumerate(symbols[i:], start=i):
            pair = returns[[left, right]].dropna()
            overlap.loc[left, right] = overlap.loc[right, left] = len(pair)
            if left == right:
                distance.loc[left, right] = 0.0
                continue
            if len(pair) < MIN_OVERLAP_DAYS:
                continue

            rho = _safe_corr(pair[left].to_numpy(), pair[right].to_numpy())
            # rho=1 -> 0; rho=0 -> sqrt(2)/2; rho=-1 -> 1  # 将相关性转化为距离
            corr_distance = np.sqrt(max(0.0, 2.0 * (1.0 - rho))) / 2.0
            # 找出两个品种都有效的指纹维度
            valid = np.isfinite(fp_scaled[i]) & np.isfinite(fp_scaled[j])
            if not valid.any():
                continue

            # 计算两个品种的原始指纹距离 -- 欧氏距离
            raw_fp_distance = float(
                np.linalg.norm(fp_scaled[i, valid] - fp_scaled[j, valid]) /
                np.sqrt(valid.sum()))
            # 将指纹距离压缩到 [0,1)
            fp_distance = (raw_fp_distance / (1.0 + raw_fp_distance))

            value = (CORR_WEIGHT * corr_distance +
                     FINGERPRINT_WEIGHT * fp_distance)
            distance.loc[left, right] = distance.loc[right, left] = value
    return distance, overlap


def score_candidate(selected, candidate, distance, anchor_distance):
    pair_distances = np.array(
        [distance.loc[candidate, member] for member in selected], dtype=float)
    if not np.isfinite(pair_distances).all():
        return {
            "candidate": candidate,
            "avg_distance": np.inf,
            "max_distance": np.inf,
            "avg_ratio": np.inf,
            "max_ratio": np.inf,
        }

    avg_distance = float(pair_distances.mean())
    max_distance = float(pair_distances.max())
    return {
        "candidate": candidate,
        "avg_distance": avg_distance,
        "max_distance": max_distance,
        "avg_ratio": avg_distance / anchor_distance,
        "max_ratio": max_distance / anchor_distance,
    }


def anchor_core_distance(core, distance) -> float:
    """计算 A 级最小核内部平均距离，扩组全程固定不变。"""
    values = [
        float(distance.loc[a, b]) for i, a in enumerate(core)
        for b in core[i + 1:] if np.isfinite(distance.loc[a, b])
    ]
    if not values:
        raise ValueError("最小核没有足够的共同历史，无法计算锚定距离")
    value = float(np.mean(values))
    if value < EPS:
        raise ValueError("最小核距离过小，不适合作为相对倍数标尺")
    return value


def expand_minimum_core(distance, core, candidates):
    """
    每轮只加入一个与当前组最近且通过倍数阈值的品种。
    分母始终是原始 A 级最小核距离，不随扩组放宽。
    """
    selected = list(dict.fromkeys(core))
    remaining = [x for x in dict.fromkeys(candidates) if x not in selected]
    anchor_distance = anchor_core_distance(core, distance)
    history = []
    round_no = 1
    while remaining:
        scores = [
            score_candidate(selected, c, distance, anchor_distance)
            for c in remaining
        ]
        valid = [
            x for x in scores if x["avg_ratio"] <= MAX_AVG_RATIO
            and x["max_ratio"] <= MAX_PAIR_RATIO
        ]
        if not valid:
            for row in scores:
                history.append({
                    "round": round_no,
                    "selected_before": ",".join(selected),
                    **row,
                    "accepted": False,
                    "reason": "no_candidate_passed_thresholds",
                })
            break
        best = min(valid, key=lambda x: (x["avg_ratio"], x["max_ratio"]))
        for row in scores:
            accepted = row["candidate"] == best["candidate"]
            history.append({
                "round":
                round_no,
                "selected_before":
                ",".join(selected),
                **row,
                "accepted":
                accepted,
                "reason":
                "best_valid_candidate" if accepted else "not_best_this_round",
            })
        selected.append(str(best["candidate"]))
        remaining.remove(str(best["candidate"]))
        round_no += 1

    return selected, pd.DataFrame(history)


def rolling_inclusion_rate(returns,
                           core,
                           candidates,
                           window_days: int = 756,
                           step_days: int = 252):
    """在多个时间窗口重复扩组，统计品种入组率。"""
    all_symbols = list(dict.fromkeys([*core, *candidates]))
    rows = []
    if len(returns) < window_days:
        window_days = len(returns)

    starts = list(range(0, max(1, len(returns) - window_days + 1), step_days))
    last_start = max(0, len(returns) - window_days)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    for start in sorted(set(starts)):
        sample = returns.iloc[start:start + window_days]
        dist, _ = build_pairwise_distance(sample[all_symbols])
        selected, _ = expand_minimum_core(dist, core, candidates)
        rows.append({
            "start": sample.index.min(),
            "end": sample.index.max(),
            "selected": ",".join(selected),
            **{
                symbol: int(symbol in selected)
                for symbol in all_symbols
            },
        })

    detail = pd.DataFrame(rows)
    rates = detail[all_symbols].mean().sort_values(ascending=False)
    rates.name = "inclusion_rate"
    return rates, detail


def close_to_daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """close: index=交易日, columns=品种。"""
    close = close.sort_index().apply(pd.to_numeric, errors="coerce")
    returns = np.log(close / close.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan)


def start(begin_date, end_date):
    core_codes = ['RB', 'HC']
    candidates_codes = ['I', 'J', 'JM', 'SF', 'SM']
    symbols = list(dict.fromkeys([*core_codes, *candidates_codes]))
    market_data = fetch_daily_market(begin_date=begin_date,
                                     end_date=end_date,
                                     codes=symbols)
    wide_market = market_data.set_index(['trade_date', 'code']).unstack()
    returns = close_to_daily_returns(wide_market['close'][symbols])
    distance, overlap = build_pairwise_distance(returns=returns)

    selected, history = expand_minimum_core(distance, core_codes,
                                            candidates_codes)
    rates, rolling = rolling_inclusion_rate(returns=returns,
                                            core=core_codes,
                                            candidates=candidates_codes,
                                            window_days=WINDOW_DAYS,
                                            step_days=STEP_DAYS)

    print("\n=== 综合距离矩阵（0=相同，1=很远） ===")
    print(distance.round(3).to_string())
    print("\n=== 全样本扩组结果 ===")
    print(f"固定 A 级核心距离: {anchor_core_distance(core_codes, distance):.3f}")
    print(" -> ".join(core_codes) + " => " + ", ".join(selected))
    print("\n=== 每轮候选评分 ===")
    columns = [
        "round",
        "selected_before",
        "candidate",
        "avg_distance",
        "max_distance",
        "avg_ratio",
        "max_ratio",
        "accepted",
        "reason",
    ]
    print(history[columns].round(3).to_string(index=False))
    print("\n=== 滚动窗口入组率 ===")
    print(rates.round(3).to_string())
    stable = rates[rates >= MIN_INCLUSION_RATE].index.tolist()
    print(f"\n固定组（入组率 >= {MIN_INCLUSION_RATE:.0%}）: " + ", ".join(stable))
    print("\n=== 滚动窗口细节 ===")
    print(rolling[["start", "end", "selected"]].to_string(index=False))
    print("\n=== 最小重叠天数 ===")
    print(overlap.to_string())


if __name__ == '__main__':
    start(begin_date='2015-01-01', end_date='2026-08-01')
