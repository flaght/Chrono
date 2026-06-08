#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于“19个基准特征共性 + 去冗余”筛选新增候选特征。

用途:
1) 从 ims1.csv 中读取候选因子指标
2) 按质量分 + 与基准特征相似度 + 家族多样性筛出:
   - 新增候选 (add)
   - 可替换候选 (replace)
3) 产出 csv + 便于直接粘贴到配置的 txt

仅生成候选名单，不训练、不回测。
"""

from __future__ import annotations
import pdb
import argparse
import csv
import json
import math
import os
import re
import statistics
from typing import Dict, List, Tuple


BASE_FEATURES_19 = [
    "MSUM(120,MDEMA(90,MCPS(90,'high')))",
    "MDEMA(120,MCPS(120,WMA(90,'twap')))",
    "MMAX(120,MDEMA(60,MCPS(90,'low')))",
    "MMASSI(120,MPRO(60,MVHF(10,'money')),MAPOSITIVE(10,'twap'))",
    "MDEMA(120,MCPS(120,MADecay(60,'twap')))",
    "MT3(120,MCPS(30,'close'))",
    "MA(60,RSI(120,MCPS(120,MA(60,'twap'))))",
    "MT3(120,MCPS(60,'high'))",
    "DELTA(90,MMIN(15,MHMA(90,DELTA(90,'close'))))/MDIFF(90,'close')",
    "MCPS(120,MT3(90,MMaxDiff(120,'twap')))",
    "MADecay(5,MMASSI(120,MT3(5,'corr_vwap_bid_size_0'),'twap'))",
    "MMeanRes(120,'corr_money_bid_size_0','smart_tick_in_pct')",
    "WMA(30,MMedian(90,'smart_tick_in_pct'))",
    "MMAX(15,MDPO(240,EMA(90,'smart_money_in_pct')))",
    "RSI(120,MCPS(120,EMA(120,'close')))",
    "MSUM(120,MDEMA(90,MCPS(90,'low')))",
    "MDIFF(90,MMeanRes(120,'corr_money_bid_size_0','smart_tick_in_pct'))",
    "MSUM(5,MADecay(10,MMedian(90,'smart_tick_in_pct')))",
    "MMedian(90,MADecay(10,MT3(5,'smart_tick_in_pct')))",
]


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|'[^']*'")
FUNC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def normalize_expr(expr: str) -> str:
    expr = (expr or "").strip()
    expr = re.sub(r"\s+", "", expr)
    return expr


def extract_tokens(expr: str) -> List[str]:
    tokens = TOKEN_RE.findall(expr)
    out = []
    for t in tokens:
        if t.startswith("'") and t.endswith("'"):
            out.append(t.lower())
        else:
            out.append(t.upper())
    return out


def token_set(expr: str) -> set:
    return set(extract_tokens(expr))


def family(expr: str) -> str:
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr or "")
    return m.group(1).upper() if m else "UNKNOWN"


def first_funcs(expr: str, k: int = 3) -> Tuple[str, ...]:
    funcs = [x.upper() for x in FUNC_RE.findall(expr or "")]
    if not funcs:
        return ("UNKNOWN",)
    return tuple(funcs[:k])


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def to_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def robust_z(values: List[float]) -> List[float]:
    if not values:
        return []
    med = statistics.median(values)
    abs_dev = [abs(x - med) for x in values]
    mad = statistics.median(abs_dev)
    scale = mad * 1.4826
    if scale < 1e-12:
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
        std = math.sqrt(var)
        if std < 1e-12:
            return [0.0 for _ in values]
        return [(x - mean) / std for x in values]
    return [(x - med) / scale for x in values]


def load_rows(csv_path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            expr = normalize_expr(r.get("expression", ""))
            if not expr:
                continue
            rows.append(
                {
                    "expression": expr,
                    "category": (r.get("category", "") or "").strip().lower(),
                    "ann_sharpe": to_float(r.get("ann_sharpe"), 0.0),
                    "calmar": to_float(r.get("calmar"), 0.0),
                    "ic_mean": to_float(r.get("ic_mean"), 0.0),
                    "icir": to_float(r.get("icir"), 0.0),
                    "pl_ratio": to_float(r.get("pl_ratio"), 0.0),
                    "turnover": to_float(r.get("turnover"), 0.0),
                    "factor_ac": to_float(r.get("factor_ac"), 0.0),
                    "ret_ac": to_float(r.get("ret_ac"), 0.0),
                    "token_set": token_set(expr),
                    "family": family(expr),
                    "func_sig": first_funcs(expr, k=3),
                }
            )
    return rows


def score_rows(cands: List[Dict]) -> None:
    if not cands:
        return
    cols = {
        "ann_sharpe": [x["ann_sharpe"] for x in cands],
        "calmar": [x["calmar"] for x in cands],
        "ic_mean": [x["ic_mean"] for x in cands],
        "icir": [x["icir"] for x in cands],
        "pl_ratio": [x["pl_ratio"] for x in cands],
        "turnover": [x["turnover"] for x in cands],
        "abs_factor_ac": [abs(x["factor_ac"]) for x in cands],
    }
    z_ann = robust_z(cols["ann_sharpe"])
    z_cal = robust_z(cols["calmar"])
    z_icm = robust_z(cols["ic_mean"])
    z_icir = robust_z(cols["icir"])
    z_pl = robust_z(cols["pl_ratio"])
    z_turn = robust_z(cols["turnover"])
    z_fac = robust_z(cols["abs_factor_ac"])

    for i, row in enumerate(cands):
        quality = (
            0.34 * z_ann[i]
            + 0.23 * z_cal[i]
            + 0.20 * z_icm[i]
            + 0.13 * z_icir[i]
            + 0.10 * z_pl[i]
        )
        stability = -0.10 * z_turn[i] - 0.10 * z_fac[i]
        cat_bonus = 0.12 if row["category"] == "p" else 0.0
        row["score"] = quality + stability + cat_bonus


def compute_similarity_to_base(rows: List[Dict], base_exprs: List[str]) -> None:
    base_norm = [normalize_expr(x) for x in base_exprs]
    base_sets = [token_set(x) for x in base_norm]

    for row in rows:
        sims = [jaccard(row["token_set"], b) for b in base_sets]
        if sims:
            i_max = max(range(len(sims)), key=lambda i: sims[i])
            row["max_sim"] = sims[i_max]
            row["closest_base"] = base_norm[i_max]
        else:
            row["max_sim"] = 0.0
            row["closest_base"] = ""


def select_add_pool(
    rows: List[Dict],
    base_exprs: List[str],
    sim_min: float,
    sim_max: float,
    duplicate_sim: float,
) -> List[Dict]:
    base_set = set(normalize_expr(x) for x in base_exprs)
    pool = []
    for r in rows:
        if r["expression"] in base_set:
            continue
        if r["max_sim"] >= duplicate_sim:
            continue
        if not (sim_min <= r["max_sim"] <= sim_max):
            continue
        pool.append(r)
    return pool


def diverse_pick(
    pool: List[Dict],
    topk: int,
    max_per_family: int,
    pairwise_sim_max: float,
) -> List[Dict]:
    picked: List[Dict] = []
    fam_cnt: Dict[str, int] = {}
    sorted_pool = sorted(pool, key=lambda x: x["score"], reverse=True)
    for cand in sorted_pool:
        fam = cand["family"]
        if fam_cnt.get(fam, 0) >= max_per_family:
            continue
        ok = True
        for p in picked:
            s = jaccard(cand["token_set"], p["token_set"])
            if s > pairwise_sim_max:
                ok = False
                break
        if not ok:
            continue
        picked.append(cand)
        fam_cnt[fam] = fam_cnt.get(fam, 0) + 1
        if len(picked) >= topk:
            break
    return picked


def select_replace_pool(
    rows: List[Dict],
    base_exprs: List[str],
    sim_low: float,
    sim_high: float,
    topk: int,
) -> List[Dict]:
    base_set = set(normalize_expr(x) for x in base_exprs)
    pool = []
    for r in rows:
        if r["expression"] in base_set:
            continue
        if sim_low <= r["max_sim"] < sim_high:
            pool.append(r)
    pool = sorted(pool, key=lambda x: x["score"], reverse=True)
    return pool[:topk]


def write_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "expression",
        "category",
        "score",
        "ann_sharpe",
        "calmar",
        "ic_mean",
        "icir",
        "pl_ratio",
        "turnover",
        "factor_ac",
        "max_sim",
        "closest_base",
        "family",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fields}
            w.writerow(out)


def write_txt_feature_list(path: str, rows: List[Dict], title: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        for r in rows:
            f.write(f"- \"{r['expression']}\"\n")


def main():
    parser = argparse.ArgumentParser(description="Feature candidate selector")
    parser.add_argument("--csv", required=True, help="ims1.csv path")
    parser.add_argument("--out_dir", required=True, help="output directory")
    parser.add_argument("--topk_add", type=int, default=8)
    parser.add_argument("--topk_replace", type=int, default=6)
    parser.add_argument("--sim_min", type=float, default=0.20)
    parser.add_argument("--sim_max", type=float, default=0.68)
    parser.add_argument("--duplicate_sim", type=float, default=0.88)
    parser.add_argument("--replace_sim_low", type=float, default=0.68)
    parser.add_argument("--replace_sim_high", type=float, default=0.88)
    parser.add_argument("--max_per_family", type=int, default=2)
    parser.add_argument("--pairwise_sim_max", type=float, default=0.58)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise ValueError("No valid rows loaded from csv")
    pdb.set_trace()
    compute_similarity_to_base(rows, BASE_FEATURES_19)
    score_rows(rows)

    add_pool = select_add_pool(
        rows=rows,
        base_exprs=BASE_FEATURES_19,
        sim_min=args.sim_min,
        sim_max=args.sim_max,
        duplicate_sim=args.duplicate_sim,
    )
    pdb.set_trace()
    add_pick = diverse_pick(
        pool=add_pool,
        topk=args.topk_add,
        max_per_family=args.max_per_family,
        pairwise_sim_max=args.pairwise_sim_max,
    )

    replace_pick = select_replace_pool(
        rows=rows,
        base_exprs=BASE_FEATURES_19,
        sim_low=args.replace_sim_low,
        sim_high=args.replace_sim_high,
        topk=args.topk_replace,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    add_csv = os.path.join(args.out_dir, "feature_shortlist_add.csv")
    rep_csv = os.path.join(args.out_dir, "feature_shortlist_replace.csv")
    add_txt = os.path.join(args.out_dir, "feature_shortlist_add.txt")
    rep_txt = os.path.join(args.out_dir, "feature_shortlist_replace.txt")
    add_json = os.path.join(args.out_dir, "feature_shortlist_add.json")

    write_csv(add_csv, add_pick)
    write_csv(rep_csv, replace_pick)
    write_txt_feature_list(add_txt, add_pick, "ADD candidates")
    write_txt_feature_list(rep_txt, replace_pick, "REPLACE candidates")

    with open(add_json, "w", encoding="utf-8") as f:
        json.dump([x["expression"] for x in add_pick], f, ensure_ascii=False, indent=2)

    print("== Done ==")
    print(f"add_csv      : {add_csv}")
    print(f"replace_csv  : {rep_csv}")
    print(f"add_txt      : {add_txt}")
    print(f"replace_txt  : {rep_txt}")
    print(f"add_json     : {add_json}")
    print("\n[ADD shortlist]")
    for i, r in enumerate(add_pick, 1):
        print(
            f"{i:02d}. score={r['score']:.4f} sim={r['max_sim']:.3f} "
            f"cat={r['category']} fam={r['family']} | {r['expression']}"
        )
    print("\n[REPLACE shortlist]")
    for i, r in enumerate(replace_pick, 1):
        print(
            f"{i:02d}. score={r['score']:.4f} sim={r['max_sim']:.3f} "
            f"closest={r['closest_base']} | {r['expression']}"
        )


if __name__ == "__main__":
    main()

