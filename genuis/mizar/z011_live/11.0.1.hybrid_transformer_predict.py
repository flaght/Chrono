"""原始分钟 bar 严格逐根计算特征 -> HybridTransformer 三 seed 预测/信号。

11.0.2 保留已计算特征的快速回放。本入口读取 OHLCV/value/vwap/openint，
方向文件必须是训练时的 formula,direction CSV 或 JSON。文件可一次读入内存，
但特征和模型每次只收到当前这一根 bar；绝不预计算整个测试区间的特征。
真实逐 bar 接口见 ``lib.nn005.live_features.LiveRawBarPredictor``。
"""
import pdb
import argparse
import time
from pathlib import Path

import pandas as pd

from lib.nn005.live import load_model, seed_model_dirs
from lib.nn005.live_features import (BAR_COLUMNS, LiveRawBarPredictor,
                                     load_feature_directions)
from lib.nn005.live_signal import SIGNAL_PRESETS, LiveSignalConverter
from lib.nn005.live_validation import (compare_prepared_features,
                                       compare_saved_predictions,
                                       compare_signal_results,
                                       load_research_signals,
                                       load_saved_research_signals)


def _read_bars(path: str, code: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".feather", ".arrow"):
        frame = pd.read_feather(path, columns=list(BAR_COLUMNS))
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, usecols=list(BAR_COLUMNS))
    else:
        raise ValueError(f"只支持 .feather/.arrow/.csv 原始 bar 文件: {path}")
    frame = frame.loc[frame["code"].astype(str).str.upper() == code,
                      list(BAR_COLUMNS)].copy()
    if frame.empty:
        raise ValueError(f"{path} 中没有 {code} 的 bar")
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="raise")
    return frame.sort_values("trade_time").reset_index(drop=True)


def _write_new_csv(frame: pd.DataFrame, path: str, label: str) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"{label}已存在，不覆盖: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"[{label}] rows={len(frame)} path={destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="从原始分钟 bar 回放三 seed 融合预测")
    parser.add_argument("--release-path", required=True,
                        help="包含 <seed>/<run_id>/config.json 的发布目录")
    parser.add_argument("--bars-path", required=True,
                        help="目标区间的原始分钟 bar Feather/CSV")
    parser.add_argument("--history-bars-path",
                        help="可选：目标区间之前的原始 bar；用于因子/模型预热")
    parser.add_argument("--directions-file", required=True,
                        help="训练时每个表达式的 formula,direction 文件")
    parser.add_argument("--code", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--feature-history-bars", type=int,
                        help="每次计算特征仅保留最近 N 根原始 bar；默认使用全部历史。"
                             "截断可能改变长窗口/EMA 因子，必须对照研究特征验证")
    parser.add_argument("--burn-in-bars", type=int, default=0,
                        help="目标区间开头 N 根原始 bar 只用于预热；不输出或比较预测")
    parser.add_argument("--max-bars", type=int,
                        help="仅处理目标文件开头 N 根 bar，包含 burn-in 区间")
    parser.add_argument("--progress-bars", type=int, default=250,
                        help="严格逐 bar 回放时每 N 根打印一次进度")
    parser.add_argument("--signal-id", choices=("none", *SIGNAL_PRESETS),
                        default="none")
    parser.add_argument("--prediction-output")
    parser.add_argument("--signal-output")
    parser.add_argument("--reference-features",
                        help="研究用 test_data.feather；核对原始 bar 特征流水线")
    parser.add_argument("--prediction-dir",
                        help="研究 ensemble/test 目录；核对各 seed 与融合预测")
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--reference-predictions",
        help="研究 ensemble_equal_predictions.csv；旧流程会重新计算参考信号")
    reference_group.add_argument(
        "--reference-signals",
        help="研究 holdout 的 signals.csv；直接核对回测实际使用的信号")
    parser.add_argument("--signal-compare-output")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    code = args.code.strip().upper()
    if args.max_bars is not None and args.max_bars <= 0:
        parser.error("--max-bars 必须大于0")
    if args.burn_in_bars < 0:
        parser.error("--burn-in-bars 不能为负数")
    if args.feature_history_bars is not None and args.feature_history_bars < 1:
        parser.error("--feature-history-bars 必须大于0")
    if args.progress_bars <= 0:
        parser.error("--progress-bars 必须大于0")
    if args.signal_output and args.signal_id == "none":
        parser.error("--signal-output 必须指定 --signal-id")
    if (args.reference_predictions or args.reference_signals) and args.signal_id == "none":
        parser.error("信号比较必须指定 --signal-id")
    if args.signal_compare_output and not (
            args.reference_predictions or args.reference_signals):
        parser.error("--signal-compare-output 需要研究预测或研究信号文件")
    for label, path in (("预测", args.prediction_output),
                        ("信号", args.signal_output),
                        ("信号比较", args.signal_compare_output)):
        if path and Path(path).exists():
            parser.error(f"{label}输出文件已存在，不覆盖: {path}")
    outputs = [Path(path).resolve() for path in
               (args.prediction_output, args.signal_output,
                args.signal_compare_output) if path]
    if len(outputs) != len(set(outputs)):
        parser.error("不同输出必须使用不同文件")

    predictor = load_model(seed_model_dirs(args.release_path),
                           requested=args.device, code=code,
                           min_history=args.min_history)
    research_signals = None
    if args.reference_signals:
        research_signals = load_saved_research_signals(
            args.reference_signals, code, args.signal_id,
            expected_run_ids=predictor.run_ids)
    elif args.reference_predictions:
        research_signals = load_research_signals(
            args.reference_predictions, code, args.signal_id,
            expected_run_ids=predictor.run_ids)
    directions = load_feature_directions(args.directions_file,
                                         predictor.features)
    live = LiveRawBarPredictor(
        predictor, directions, min_raw_bars=1,
        max_raw_history_bars=args.feature_history_bars)
    if args.feature_history_bars is not None:
        print(f"[FEATURE_HISTORY] 仅保留最近 {args.feature_history_bars} 根原始 bar；"
              "此模式可能与完整历史特征不一致，请检查 FEATURE_COMPARISON")
    target = _read_bars(args.bars_path, code)
    if args.max_bars is not None:
        target = target.head(args.max_bars).copy()
    if args.burn_in_bars >= len(target):
        parser.error("--burn-in-bars 必须小于目标区间的 bar 数")
    if args.history_bars_path:
        history = _read_bars(args.history_bars_path, code)
        if history["trade_time"].max() >= target["trade_time"].min():
            raise ValueError("预热 bar 必须全部早于目标区间，不能重叠")
        combined = pd.concat([history, target], ignore_index=True)
        history_count = len(history)
    else:
        combined = target
        history_count = 0
        if args.burn_in_bars == 0:
            print("[WARN] 没有预热 bar，也没有 burn-in：早期特征不能保证与研究数据一致")
    compare_start = history_count + args.burn_in_bars
    if args.burn_in_bars:
        print(f"[BURN_IN] skipped={args.burn_in_bars} "
              f"from={target.trade_time.iloc[0]} "
              f"through={target.trade_time.iloc[args.burn_in_bars - 1]} "
              f"first_compared={target.trade_time.iloc[args.burn_in_bars]}")
    full_coverage = args.max_bars is None and args.burn_in_bars == 0
    if len(combined) > 3000 and args.feature_history_bars is None:
        print("[WARN] 严格逐 bar 特征计算会反复处理已到达的历史；"
              "全量回放可能很慢，建议先用 --max-bars 做小样本验收")
    live.reset()
    signal_converter = (None if args.signal_id == "none" else
                        LiveSignalConverter(args.signal_id))
    records = []
    signal_records = []
    feature_records = []
    started = time.perf_counter()
    for index, row in enumerate(combined.itertuples(index=False, name=None)):
        bar = dict(zip(BAR_COLUMNS, row))
        if index < compare_start:
            live.push_bar(bar)
        else:
            pdb.set_trace()
            result = live.predict_bar(bar)
            if result is not None:
                result["sample_block"] = 1
                # 实盘顺序：当前 bar 预测后立即转信号，再接收下一根 bar。
                if signal_converter is not None:
                    signal = signal_converter.convert(result)
                    signal["eval_5m_mark"] = (
                        pd.Timestamp(signal["trade_time"]).minute % 5 == 0)
                    signal_records.append(signal)
                records.append(result)
                if args.reference_features:
                    feature_records.append(live.last_features)
        if (index + 1) % args.progress_bars == 0 or index + 1 == len(combined):
            print(f"[STREAM_PROGRESS] bars={index + 1}/{len(combined)} "
                  f"predictions={len(records)} "
                  f"elapsed={time.perf_counter() - started:.1f}s")
    predictions = pd.DataFrame.from_records(records)
    if predictions.empty:
        raise ValueError("没有可输出的预测，请检查 --min-history 和原始 bar")
    if args.reference_features:
        reference = pd.read_feather(args.reference_features,
                                    columns=["trade_time", "code",
                                             *predictor.features])
        report = compare_prepared_features(
            pd.DataFrame.from_records(feature_records), reference,
            predictor.features, code=code, atol=args.atol, rtol=args.rtol,
            require_full_coverage=full_coverage)
        print("[FEATURE_COMPARISON]")
        print(report.to_string(index=False))
    print(f"code={code} seeds={list(predictor.keys())} "
          f"lookback={predictor.lookback} predictions={len(predictions)} "
          f"device={predictor.device}")
    print(predictions.tail().to_string(index=False))
    if args.prediction_output:
        _write_new_csv(predictions, args.prediction_output, "PREDICTION_OUTPUT")
    signals = (None if signal_converter is None else
               pd.DataFrame.from_records(signal_records))
    if signals is not None:
        print("[LIVE_SIGNALS]")
        print(signals[["trade_time", "code", "predicted_z", "signal",
                       "signal_id", "eval_5m_mark"]].tail().to_string(index=False))
    if args.signal_output:
        _write_new_csv(signals, args.signal_output, "SIGNAL_OUTPUT")
    if research_signals is not None:
        summary, details = compare_signal_results(
            signals, research_signals, z_atol=args.atol, z_rtol=args.rtol,
            require_full_coverage=full_coverage)
        print("[RESEARCH_SIGNAL_COMPARISON]")
        print(summary.to_string(index=False))
        if args.signal_compare_output:
            _write_new_csv(details, args.signal_compare_output,
                           "SIGNAL_COMPARISON_OUTPUT")
    if args.prediction_dir:
        report = compare_saved_predictions(
            predictions, predictor, args.prediction_dir,
            atol=args.atol, rtol=args.rtol,
            require_full_coverage=full_coverage)
        print("[RESEARCH_PREDICTION_COMPARISON]")
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
