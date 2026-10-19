"""逐 bar 回放、抽样及与离线预测 CSV 的一致性校验；不用于实盘路径。"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from .live import LiveEnsemblePredictor, load_model, seed_model_dirs
from .live_signal import SIGNAL_PRESETS


def compare_prepared_features(live, reference, features, code: str,
                              atol: float = 1e-5, rtol: float = 1e-5,
                              require_full_coverage: bool = True):
    """按 trade_time/code 核对原始 bar 重建特征与研究特征文件。"""
    import pandas as pd

    keys = ["trade_time", "code"]
    columns = keys + list(features)
    for name, frame in (("live", live), ("reference", reference)):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} 缺少特征字段: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} 特征数据为空")
    left = live[columns].copy()
    right = reference[columns].copy()
    for frame in (left, right):
        frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="raise")
        frame["code"] = frame["code"].astype(str).str.upper()
        frame.drop(frame.loc[frame["code"] != code.upper()].index,
                   inplace=True)
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("特征数据的 trade_time/code 缺失或重复")
    merged = left.merge(right, on=keys, how="outer", suffixes=("_live", "_ref"),
                        validate="one_to_one", indicator=True)
    extra = int((merged["_merge"] == "left_only").sum())
    absent = int((merged["_merge"] == "right_only").sum())
    if extra or (require_full_coverage and absent):
        raise ValueError(f"特征时间键不一致: 逐 bar 独有 {extra} 行、"
                         f"研究数据独有 {absent} 行")
    merged = merged.loc[merged["_merge"] == "both"]
    if merged.empty:
        raise ValueError("没有可比较的特征时间键")
    reports = []
    for feature in features:
        x = merged[f"{feature}_live"].to_numpy(dtype=float)
        y = merged[f"{feature}_ref"].to_numpy(dtype=float)
        delta = np.abs(x - y)
        valid = np.isfinite(x) & np.isfinite(y)
        ok = valid & np.isclose(x, y, atol=atol, rtol=rtol)
        reports.append({
            "code": code.upper(), "feature": feature,
            "compared_rows": len(merged),
            "mismatch_rows": int((~ok).sum()),
            "max_abs_error": float(np.max(delta)) if valid.all() else np.inf,
            "mean_abs_error": float(np.mean(delta)) if valid.all() else np.inf,
            "passed": bool(ok.all()),
        })
    return pd.DataFrame.from_records(reports)


def _sample_intervals(total: int, max_bars: int, mode: str,
                      blocks: int, seed: int) -> list[tuple[int, int]]:
    """随机模式在不同时间分区各取一段，避免重叠或只抽中同一时期。"""
    if mode not in ("head", "tail", "random"):
        raise ValueError("sample_mode 必须是 head、tail 或 random")
    if int(blocks) <= 0 or (mode != "random" and int(blocks) != 1):
        raise ValueError("sample_blocks 仅在 random 模式下可大于1")
    if max_bars is None:
        if mode != "head" or int(blocks) != 1:
            raise ValueError("tail/random 模式必须指定 max_bars")
        return [(0, total)]
    if int(max_bars) <= 0:
        raise ValueError("max_bars 必须大于0")
    width = min(int(max_bars), total)
    if mode == "head":
        return [(0, width)]
    if mode == "tail":
        return [(total - width, total)]
    if width * int(blocks) > total:
        raise ValueError("random 模式的 max_bars × sample_blocks 不能超过总 bar 数")
    rng = np.random.default_rng(int(seed))
    intervals = []
    for number in range(int(blocks)):
        left = total * number // int(blocks)
        right = total * (number + 1) // int(blocks)
        start = int(rng.integers(left, right - width + 1))
        intervals.append((start, start + width))
    return intervals


def replay_data1(data1, predictor: LiveEnsemblePredictor,
                 warmup_bars: int = 10, max_bars: int = None,
                 sample_mode: str = "head", sample_blocks: int = 1,
                 sample_seed: int = 42):
    """用 DataFrame 模拟逐 bar 推送；抽样段前补足历史，不对未抽中的行推理。"""
    import pandas as pd

    if int(warmup_bars) < 0:
        raise ValueError("warmup_bars 不能为负数")
    columns = ["trade_time", "code", *predictor.features]
    missing = set(columns) - set(data1.columns)
    if missing:
        raise ValueError(f"data1 缺少字段: {sorted(missing)}")
    frame = data1.loc[data1["code"].astype(str).str.upper() == predictor.code,
                      columns].sort_values("trade_time")
    if frame.empty:
        raise ValueError(f"data1 没有 {predictor.code} 的 bar")
    intervals = _sample_intervals(len(frame), max_bars, sample_mode,
                                  sample_blocks, sample_seed)
    if any(end - start <= warmup_bars for start, end in intervals):
        raise ValueError("每段抽样长度必须大于 warmup_bars；比对离线预测请设为0")
    records = []
    for block_number, (start, end) in enumerate(intervals, start=1):
        predictor.reset()
        # SequenceStore 的窗口最多只需要前 lookback-1 根历史 bar。
        # 中间若有交易时段断点，push_values 会自动清空旧片段。
        context_start = max(0, start - predictor.lookback + 1)
        for row in frame.iloc[context_start:start].itertuples(index=False,
                                                               name=None):
            predictor.push_values(row[0], row[1], row[2:])
        for number, row in enumerate(frame.iloc[start:end].itertuples(
                index=False, name=None)):
            if number < warmup_bars:
                predictor.push_values(row[0], row[1], row[2:])
                continue
            result = predictor.predict_values(row[0], row[1], row[2:])
            if result is not None:
                result["sample_block"] = block_number
                records.append(result)
    return pd.DataFrame.from_records(records)


def compare_saved_predictions(results, predictor: LiveEnsemblePredictor,
                              prediction_dir: str, atol: float = 1e-5,
                              rtol: float = 1e-5,
                              require_full_coverage: bool = True):
    """按品种和时间比较逐 bar 与离线 CSV 的每个 seed 及等权融合预测。"""
    import pandas as pd

    root = Path(prediction_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"离线预测目录不存在: {root}")
    if results.empty or atol < 0 or rtol < 0:
        raise ValueError("预测结果不能为空，atol/rtol 不能为负")
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected_ids = {int(seed): Path(path).name for seed, path
                        in manifest["run_dirs"].items()}
        if expected_ids != predictor.run_ids:
            raise ValueError(f"离线与实盘 run_id 不一致: {expected_ids} != "
                             f"{predictor.run_ids}")

    required = {"trade_time", "code", "predicted_z", "predicted_ret_h",
                "per_seed_predicted_z"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"逐 bar 结果缺少字段: {sorted(missing)}")
    live = results[["trade_time", "code", "predicted_z",
                    "predicted_ret_h", "per_seed_predicted_z"]].copy()
    live["trade_time"] = pd.to_datetime(live["trade_time"], errors="raise")
    live["code"] = live["code"].astype(str).str.upper()
    if live["code"].isna().any() or set(live["code"]) != {predictor.code}:
        raise ValueError("逐 bar 结果品种与预测器不一致")
    keys = ["trade_time", "code"]
    if live[keys].isna().any().any() or live.duplicated(keys).any():
        raise ValueError("逐 bar 结果的 trade_time/code 缺失或重复")
    for seed in predictor.models:
        try:
            live[f"seed_{seed}"] = live["per_seed_predicted_z"].map(
                lambda values: values[seed])
        except (KeyError, TypeError) as error:
            raise ValueError(f"逐 bar 结果缺少 seed={seed} 的预测") from error

    reports = []
    schemes = [*(f"seed_{seed}" for seed in predictor.models),
               "ensemble_equal"]
    for scheme in schemes:
        path = root / f"{scheme}_predictions.csv"
        expected = pd.read_csv(path, usecols=[*keys, "predicted_z",
                                              "predicted_ret_h", "ret_scale"])
        expected["trade_time"] = pd.to_datetime(expected["trade_time"],
                                                 errors="raise")
        expected["code"] = expected["code"].astype(str).str.upper()
        expected = expected.loc[expected["code"] == predictor.code]
        if expected.empty or expected[keys].isna().any().any():
            raise ValueError(f"{path} 无 {predictor.code} 的有效预测键")
        if expected.duplicated(keys).any():
            raise ValueError(f"{path} 的 trade_time/code 重复")
        if not np.allclose(expected["ret_scale"].to_numpy(dtype=float),
                           predictor.ret_scale, rtol=0, atol=1e-12):
            raise ValueError(f"{path} 的 ret_scale 与发布模型不一致")
        joined = live.merge(expected, on=keys, how="outer", indicator=True,
                            validate="one_to_one", suffixes=("_live", "_csv"))
        extra = joined.loc[joined["_merge"] == "left_only", keys]
        missing = joined.loc[joined["_merge"] == "right_only", keys]
        if len(extra) or (require_full_coverage and len(missing)):
            raise ValueError(
                f"{scheme} 时间键不一致: 逐 bar 独有 {len(extra)} 行、"
                f"CSV 独有 {len(missing)} 行；"
                f"首个差异={pd.concat([extra, missing]).head(1).to_dict('records')}")
        joined = joined.loc[joined["_merge"] == "both"].copy()
        live_z = joined["predicted_z_live"].to_numpy(dtype=float) if (
            scheme == "ensemble_equal") else joined[scheme].to_numpy(dtype=float)
        csv_z = joined["predicted_z_csv"].to_numpy(dtype=float)
        live_ret = live_z * predictor.ret_scale
        csv_ret = joined["predicted_ret_h_csv"].to_numpy(dtype=float)
        finite = (np.isfinite(live_z) & np.isfinite(csv_z) &
                  np.isfinite(live_ret) & np.isfinite(csv_ret))
        z_ok = np.isclose(live_z, csv_z, atol=atol, rtol=rtol)
        ret_ok = np.isclose(live_ret, csv_ret,
                            atol=atol * predictor.ret_scale, rtol=rtol)
        # 同时检查实时返回的融合原始收益列，防止缩放漏乘或重复乘。
        if scheme == "ensemble_equal":
            returned_ret = joined["predicted_ret_h_live"].to_numpy(dtype=float)
            ret_ok &= np.isclose(returned_ret, csv_ret,
                                  atol=atol * predictor.ret_scale, rtol=rtol)
        ok = finite & z_ok & ret_ok
        delta = np.abs(live_z - csv_z)
        worst = int(np.nanargmax(np.where(finite, delta, np.inf))) if len(delta) else 0
        reports.append({
            "scheme": scheme, "code": predictor.code,
            "compared_rows": len(joined), "csv_rows": len(expected),
            "mismatch_rows": int((~ok).sum()),
            "max_abs_error_z": float(np.max(delta)) if finite.all() else np.inf,
            "mean_abs_error_z": float(np.mean(delta)) if finite.all() else np.inf,
            "worst_trade_time": joined.iloc[worst]["trade_time"],
            "worst_live_z": float(live_z[worst]),
            "worst_csv_z": float(csv_z[worst]),
            "passed": bool(ok.all()),
        })
    return pd.DataFrame(reports)


def _research_signal_values(predicted_z, signal_id: str):
    """独立复现研究环境的三个 Lumina 信号公式，用于校验实时实现。"""
    method, params = SIGNAL_PRESETS[str(signal_id)]
    z = np.asarray(predicted_z, dtype=float)
    if not np.isfinite(z).all():
        raise ValueError("研究预测包含非有限 predicted_z，无法比较信号")
    threshold = float(params["threshold"])
    if method == "linear_signal":
        return np.clip(z / threshold, -1.0, 1.0)
    if method == "linear_threshold_signal":
        magnitude = np.clip((np.abs(z) - threshold) /
                            (float(params["upper"]) - threshold), 0.0, 1.0)
        return np.sign(z) * magnitude
    if method == "threshold_signal":
        return (z >= threshold).astype(int) - (z <= -threshold).astype(int)
    raise ValueError(f"未实现研究信号公式: {method}")


def load_research_signals(prediction_csv: str, code: str, signal_id: str,
                          expected_run_ids=None):
    """从研究融合预测文件生成参考信号；只在此处读文件和核对模型身份。"""
    import pandas as pd

    path = Path(prediction_csv)
    if path.name != "ensemble_equal_predictions.csv" or not path.is_file():
        raise FileNotFoundError(
            f"需要研究测试集的 ensemble_equal_predictions.csv: {path}")
    signal_id = str(signal_id)
    if signal_id not in SIGNAL_PRESETS:
        raise ValueError(f"未知信号编号 {signal_id}；可选 {sorted(SIGNAL_PRESETS)}")
    code = str(code).strip().upper()
    if not code:
        raise ValueError("code 不能为空")

    manifest_path = path.parent / "manifest.json"
    if expected_run_ids is not None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"无法核对研究模型身份，缺少 {manifest_path}")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        reference_ids = {int(seed): Path(run_dir).name for seed, run_dir
                         in manifest["run_dirs"].items()}
        expected_ids = {int(seed): str(run_id) for seed, run_id
                        in expected_run_ids.items()}
        if reference_ids != expected_ids:
            raise ValueError("研究预测与当前发布模型的 seed/run_id 不一致: "
                             f"{reference_ids} != {expected_ids}")

    keys = ["trade_time", "code"]
    reference = pd.read_csv(path, usecols=[*keys, "predicted_z"])
    if reference[keys].isna().any().any():
        raise ValueError(f"{path} 的时间或品种缺失")
    reference["trade_time"] = pd.to_datetime(reference["trade_time"],
                                              errors="raise")
    reference["code"] = reference["code"].astype(str).str.upper()
    reference = reference.loc[reference["code"] == code].copy()
    if reference.empty or reference.duplicated(keys).any():
        raise ValueError(f"{path} 中 {code} 的参考预测为空或键重复")
    method = SIGNAL_PRESETS[signal_id][0]
    reference["signal"] = _research_signal_values(reference["predicted_z"],
                                                   signal_id)
    reference["signal_method"] = method
    reference["signal_id"] = signal_id
    reference["eval_5m_mark"] = reference["trade_time"].dt.minute % 5 == 0
    return reference


def load_saved_research_signals(signal_csv: str, code: str, signal_id: str,
                                expected_run_ids=None):
    """读取 holdout 实际信号，不从预测值重新计算参考信号。"""
    import pandas as pd

    path = Path(signal_csv)
    if path.name != "signals.csv" or not path.is_file():
        raise FileNotFoundError(f"需要 holdout 保存的 signals.csv: {path}")
    signal_id = str(signal_id)
    if signal_id not in SIGNAL_PRESETS:
        raise ValueError(f"未知信号编号 {signal_id}；可选 {sorted(SIGNAL_PRESETS)}")
    code = str(code).strip().upper()
    if not code:
        raise ValueError("code 不能为空")
    method = SIGNAL_PRESETS[signal_id][0]
    if expected_run_ids is not None:
        expected_ids = {int(seed): str(run_id) for seed, run_id
                        in expected_run_ids.items()}
        ensemble_name = "_".join(expected_ids[seed]
                                 for seed in sorted(expected_ids))
        holdout_dir = next((parent for parent in path.parents
                            if parent.name == "holdout"), None)
        if holdout_dir is None:
            raise ValueError(f"信号文件不在 holdout 目录下，无法核对模型身份: {path}")
        parts = path.relative_to(holdout_dir).parts
        expected_parts = (ensemble_name, "ensemble", method, signal_id,
                          f"{code}_{method}_{signal_id}_test", "signals.csv")
        if parts != expected_parts:
            raise ValueError("holdout 信号路径与发布模型/品种/信号不一致: "
                             f"{parts} != {expected_parts}")

    columns = ["trade_time", "code", "predicted_z", "signal",
               "signal_method", "signal_id"]
    try:
        reference = pd.read_csv(path, usecols=columns,
                                dtype={"signal_id": str})
    except ValueError as exc:
        raise ValueError(f"holdout 信号文件缺少必要字段: {path}: {exc}") from exc
    if reference.empty or reference[columns].isna().any().any():
        raise ValueError(f"holdout 信号为空或存在缺失字段: {path}")
    reference["trade_time"] = pd.to_datetime(reference["trade_time"],
                                              errors="raise")
    reference["code"] = reference["code"].astype(str).str.upper()
    if (set(reference["code"]) != {code}
            or set(reference["signal_id"]) != {signal_id}
            or set(reference["signal_method"]) != {method}
            or reference.duplicated(["trade_time", "code"]).any()):
        raise ValueError(f"holdout 信号的品种、编号、方法或时间键不一致: {path}")
    for column in ("predicted_z", "signal"):
        reference[column] = pd.to_numeric(reference[column], errors="raise")
        if not np.isfinite(reference[column].to_numpy(dtype=float)).all():
            raise ValueError(f"holdout 信号存在非有限 {column}: {path}")
    return reference


def compare_signal_results(live_signals, research_signals,
                           z_atol: float = 1e-5, z_rtol: float = 1e-5,
                           signal_atol: float = 1e-6,
                           signal_rtol: float = 1e-5,
                           require_full_coverage: bool = False):
    """只比较两张已生成的信号表；不读文件、不运行模型、不计算信号。"""
    import pandas as pd

    tolerances = (z_atol, z_rtol, signal_atol, signal_rtol)
    if not all(np.isfinite(value) and value >= 0 for value in tolerances):
        raise ValueError("比较容差必须为有限非负数")
    keys = ["trade_time", "code"]
    required = {*keys, "predicted_z", "signal", "signal_id", "signal_method"}
    frames = []
    for name, source in (("live", live_signals),
                         ("research", research_signals)):
        missing = required - set(source.columns)
        if missing or source.empty:
            raise ValueError(f"{name} 信号结果为空或缺少字段: {sorted(missing)}")
        frame = source.copy()
        if frame[keys].isna().any().any():
            raise ValueError(f"{name} 时间或品种缺失")
        frame["trade_time"] = pd.to_datetime(frame["trade_time"],
                                              errors="raise")
        frame["code"] = frame["code"].astype(str).str.upper()
        frame["signal_id"] = frame["signal_id"].astype(str)
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} 时间/品种键重复")
        frames.append(frame)
    live, reference = frames
    for field in ("code", "signal_id", "signal_method"):
        if live[field].nunique() != 1 or reference[field].nunique() != 1 or (
                live[field].iloc[0] != reference[field].iloc[0]):
            raise ValueError(f"两侧 {field} 不一致，不能比较")
    code = live["code"].iloc[0]
    signal_id = live["signal_id"].iloc[0]
    if require_full_coverage and len(live) != len(reference):
        raise ValueError(f"完整回放行数与研究测试集不同: "
                         f"{len(live)} != {len(reference)}")

    live_columns = [*keys, "predicted_z", "signal"]
    if "sample_block" in live.columns:
        live_columns.append("sample_block")
    details = live[live_columns].merge(
        reference[[*keys, "predicted_z", "signal"]], on=keys, how="left",
        indicator=True, validate="one_to_one", suffixes=("_live", "_research"))
    unmatched = details.loc[details["_merge"] != "both", keys]
    if not unmatched.empty:
        raise ValueError(f"研究测试集缺少 {len(unmatched)} 个回放时间键；"
                         f"首个={unmatched.iloc[0].to_dict()}")
    details = details.drop(columns="_merge")
    live_z = details["predicted_z_live"].to_numpy(dtype=float)
    reference_z = details["predicted_z_research"].to_numpy(dtype=float)
    live_signal = details["signal_live"].to_numpy(dtype=float)
    reference_signal = details["signal_research"].to_numpy(dtype=float)
    if not all(np.isfinite(values).all() for values in
               (live_z, reference_z, live_signal, reference_signal)):
        raise ValueError("预测或信号包含非有限值")
    details["abs_error_z"] = np.abs(live_z - reference_z)
    details["abs_error_signal"] = np.abs(live_signal - reference_signal)
    details["z_match"] = np.isclose(live_z, reference_z,
                                    atol=z_atol, rtol=z_rtol)
    details["signal_match"] = np.isclose(live_signal, reference_signal,
                                         atol=signal_atol, rtol=signal_rtol)
    details["signal_state_match"] = (np.sign(live_signal) ==
                                     np.sign(reference_signal))
    details["eval_5m_mark"] = details["trade_time"].dt.minute % 5 == 0

    summary = []
    for scope, part in (("all_bars", details),
                        ("eval_5m_mark", details.loc[details["eval_5m_mark"]])):
        summary.append({
            "scope": scope, "code": code, "signal_id": signal_id,
            "research_rows": len(reference), "compared_rows": len(part),
            "z_mismatch_rows": int((~part["z_match"]).sum()),
            "signal_mismatch_rows": int((~part["signal_match"]).sum()),
            "signal_state_mismatch_rows": int((~part["signal_state_match"]).sum()),
            "max_abs_error_z": (float(part["abs_error_z"].max())
                                if len(part) else float("nan")),
            "mean_abs_error_z": (float(part["abs_error_z"].mean())
                                 if len(part) else float("nan")),
            "max_abs_error_signal": (float(part["abs_error_signal"].max())
                                     if len(part) else float("nan")),
            "mean_abs_error_signal": (float(part["abs_error_signal"].mean())
                                      if len(part) else float("nan")),
        })
    return pd.DataFrame(summary), details


