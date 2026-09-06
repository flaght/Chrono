"""Polars AlphaGPT 因子挖掘运行示例。"""

import argparse
from pathlib import Path

import polars as pl

from miner.alphagpt import AlphaGPTConfig, Launcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--input", required=True, help="Feather/IPC 数据文件")
    # parser.add_argument("--features", nargs="+", required=True)
    # parser.add_argument("--return-column", required=True)
    parser.add_argument("--mode", choices=("free", "directed"), default="free")
    parser.add_argument("--unary-operators", nargs="*")
    parser.add_argument("--window-operators", nargs="*")
    parser.add_argument("--binary-operators", nargs="*")
    parser.add_argument("--pair-operators", nargs="*")
    parser.add_argument("--periods", nargs="+", type=int,
                        default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-formula-len", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--device", help="例如 cpu、cuda、cuda:0")
    parser.add_argument("--score", choices=("abs_ic_mean", "ic_sharpe"),
                        default="abs_ic_mean")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="每隔多少轮输出一次训练日志")
    parser.add_argument("--quiet", action="store_true", help="关闭训练进度日志")
    parser.add_argument("--output", help="结果 Feather/IPC 文件")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    FILE_PATH = (
        "/workspace/worker/pj/Chrono/genuis/mizar/records/"
        "ricso2/rbb/basic/recent_data.feather"
    )
    input1 = FILE_PATH
    features = ['open','high','value']
    return_column = 'close'
    config = AlphaGPTConfig(
        periods=tuple(args.periods),
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        max_formula_len=args.max_formula_len,
        device=args.device,
        score=args.score,
        log_interval=args.log_interval,
        verbose=not args.quiet,
    )
    operator_config = {
        name: value for name, value in {
            "unary": args.unary_operators,
            "window": args.window_operators,
            "binary": args.binary_operators,
            "pair": args.pair_operators,
        }.items() if value is not None
    }
    miner = Launcher(
        feature_names=features,
        return_column=return_column,
        mode=args.mode,
        config=config,
        operator_config=operator_config or None,
    )
    result = miner.optimize(pl.scan_ipc(input1), top_n=args.top_n)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_ipc(output)
    print(result)
