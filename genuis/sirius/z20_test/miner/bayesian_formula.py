
import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from miner.bayesian import BayesianConfig, Launcher


def mine(
    input_path: str,
    *,
    feature_names: list[str],
    return_column: str,
    mode: str = "free",
    operator_config: Mapping[str, Sequence[str]] = None,
    periods: tuple[int, ...] = (5, 10, 20, 40, 60),
    max_depth: int = 3,
    n_trials: int = 6000,
    n_jobs: int = 4,
    top_n: int = 150,
    output_path: str = None,
) -> pl.DataFrame:
    """运行公式挖掘；mode=free 为不定项，mode=directed 为定向。"""
    source = pl.scan_ipc(input_path)
    config = BayesianConfig(
        periods=periods,
        max_depth=max_depth,
        n_trials=n_trials,
        n_jobs=n_jobs,
        score="abs_ic_mean",
    )
    miner = Launcher(
        feature_names=feature_names,
        return_column=return_column,
        mode=mode,  # type: ignore[arg-type]
        config=config,
        operator_config=operator_config,
    )
    result = miner.optimize(source, top_n=top_n)
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.write_ipc(destination)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("free", "directed"), default="free")
    parser.add_argument("--unary-operators", nargs="*")
    parser.add_argument("--window-operators", nargs="*")
    parser.add_argument("--binary-operators", nargs="*")
    parser.add_argument("--pair-operators", nargs="*")
    parser.add_argument("--periods", nargs="+", type=int,
                        default=[5, 10, 20, 40, 60])
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=6000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=150)
    parser.add_argument("--output", help="结果 Feather/IPC 路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    FILE_PATH = (
        "/workspace/worker/pj/Chrono/genuis/mizar/records/"
        "ricso2/rbb/basic/recent_data.feather"
    )
    features = ["open", "high", "low", "close", "volume"]
    return_column = "close"
    operator_config = {
        name: value for name, value in {
            "unary": args.unary_operators,
            "window": args.window_operators,
            "binary": args.binary_operators,
            "pair": args.pair_operators,
        }.items() if value is not None
    }
    ranking = mine(
        input_path=FILE_PATH,
        feature_names=features,
        return_column=return_column,
        mode=args.mode,
        operator_config=operator_config or None,
        periods=tuple(args.periods),
        max_depth=args.max_depth,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        top_n=args.top_n,
        output_path=args.output,
    )
    print(ranking)
