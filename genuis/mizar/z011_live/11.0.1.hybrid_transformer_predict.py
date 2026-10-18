"""逐 bar 预测与离线校验的兼容调用入口。"""
import argparse
from pathlib import Path
from lib.nn005.live import LiveEnsemblePredictor, load_model, seed_model_dirs
from lib.nn005.live_validation import (
    compare_saved_predictions,
    compare_signal_results,
    load_research_signals,
    replay_data1,
)
from lib.nn005.live_signal import SIGNAL_PRESETS, generate_live_signals


def main() -> None:
    parser = argparse.ArgumentParser(description="逐 bar 回放三 seed 融合预测")
    parser.add_argument(
        "--release-path",
        default=("/workspace/worker/kdwk/Chrono/genuis/mizar/records/ricso2/"
                 "1release/pc/10001"))
    parser.add_argument(
        "--data-path",
        default=(
            "/workspace/worker/kdwk/Chrono/genuis/mizar/records/ricso2/pc/"
            "temp/model/159001/5/rl/10001/data/test_data.feather"))
    parser.add_argument("--code", default="P")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-bars", type=int, default=10)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--prediction-dir",
                        help="含三个 seed 和融合 predictions.csv 的目录")
    parser.add_argument("--max-bars", type=int, help="每段预测 N 根 bar；不指定则全量")
    parser.add_argument("--sample-mode",
                        choices=("head", "tail", "random"),
                        default="head",
                        help="取开头、结尾或随机连续区间")
    parser.add_argument("--sample-blocks",
                        type=int,
                        default=1,
                        help="random 模式抽取的区间数，分散于不同时间段")
    parser.add_argument("--sample-seed",
                        type=int,
                        default=42,
                        help="random 抽样种子，固定后可复现")
    parser.add_argument("--signal-id",
                        choices=("none", *SIGNAL_PRESETS),
                        default="none",
                        help="显式指定固定信号编号；none 只输出模型预测")
    parser.add_argument("--signal-output", help="可选：将逐 bar 预测及目标信号保存为新 CSV")
    parser.add_argument("--reference-predictions",
                        help="研究测试集的 ensemble_equal_predictions.csv；"
                        "从 predicted_z 重建参考信号，不需要旧信号文件")
    parser.add_argument("--signal-compare-output",
                        help="可选：保存研究与回放信号的逐行对比 CSV")
    parser.add_argument("--signal-atol", type=float, default=1e-6)
    parser.add_argument("--signal-rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.prediction_dir and (args.warmup_bars != 0
                                or args.min_history != 1):
        parser.error("与完整离线预测比对时请设置 --warmup-bars 0 --min-history 1")
    if args.signal_output and args.signal_id == "none":
        parser.error("保存信号时必须指定有效的 --signal-id")
    if args.reference_predictions and args.signal_id == "none":
        parser.error("比较信号时必须指定有效的 --signal-id")
    if (args.reference_predictions and args.max_bars is None and
            (args.warmup_bars != 0 or args.min_history != 1)):
        parser.error("完整测试集信号对比请设置 --warmup-bars 0 --min-history 1")
    if args.signal_compare_output and not args.reference_predictions:
        parser.error("--signal-compare-output 需要 --reference-predictions")
    if (args.signal_output and args.signal_compare_output
            and Path(args.signal_output).resolve() == Path(
                args.signal_compare_output).resolve()):
        parser.error("信号输出与对比输出必须使用不同文件")
    if args.signal_output and Path(args.signal_output).exists():
        parser.error(f"输出文件已存在，不覆盖: {args.signal_output}")
    if args.signal_compare_output and Path(
            args.signal_compare_output).exists():
        parser.error(f"对比文件已存在，不覆盖: {args.signal_compare_output}")

    import pandas as pd

    predictor = load_model(seed_model_dirs(args.release_path),
                           requested=args.device,
                           code=args.code,
                           min_history=args.min_history)
    data1 = pd.read_feather(
        args.data_path, columns=["trade_time", "code", *predictor.features])
    predictions = replay_data1(data1,
                               predictor,
                               warmup_bars=args.warmup_bars,
                               max_bars=args.max_bars,
                               sample_mode=args.sample_mode,
                               sample_blocks=args.sample_blocks,
                               sample_seed=args.sample_seed)
    print(f"code={predictor.code} seeds={list(predictor.keys())} "
          f"lookback={predictor.lookback} predictions={len(predictions)} "
          f"device={predictor.device}")
    print(predictions.tail().to_string(index=False))
    for block, part in predictions.groupby("sample_block"):
        print(f"[SAMPLE] block={block} rows={len(part)} "
              f"from={part.trade_time.iloc[0]} to={part.trade_time.iloc[-1]}")
    signals = (None if args.signal_id == "none" else
               generate_live_signals(predictions, args.signal_id))
    if signals is not None:
        print("[LIVE_SIGNALS]")
        print(signals[["trade_time", "code", "predicted_z", "signal",
                       "signal_id", "eval_5m_mark"]].tail().to_string(index=False))
    if args.signal_output:
        output_path = Path(args.signal_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "trade_time", "code", "predicted_z", "predicted_ret_h", "signal",
            "signal_method", "signal_id", "eval_5m_mark", "sample_block"
        ]
        signals[columns].to_csv(output_path, index=False)
        print(f"[SIGNAL_OUTPUT] rows={len(signals)} path={output_path}")
    if args.reference_predictions:
        research_signals = load_research_signals(
            args.reference_predictions, predictor.code, args.signal_id,
            expected_run_ids=predictor.run_ids)
        summary, details = compare_signal_results(
            signals,
            research_signals,
            z_atol=args.atol,
            z_rtol=args.rtol,
            signal_atol=args.signal_atol,
            signal_rtol=args.signal_rtol,
            require_full_coverage=args.max_bars is None)
        print("[RESEARCH_SIGNAL_COMPARISON]")
        print(summary.to_string(index=False))
        if args.signal_compare_output:
            comparison_path = Path(args.signal_compare_output)
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            details.to_csv(comparison_path, index=False)
            print(f"[SIGNAL_COMPARISON_OUTPUT] rows={len(details)} "
                  f"path={comparison_path}")
    if args.prediction_dir:
        report = compare_saved_predictions(predictions,
                                           predictor,
                                           args.prediction_dir,
                                           atol=args.atol,
                                           rtol=args.rtol,
                                           require_full_coverage=args.max_bars
                                           is None)
        print(report.to_string(index=False))
        if not report["passed"].all():
            raise SystemExit("逐 bar 与离线预测不一致；详见上表")


if __name__ == "__main__":
    main()
