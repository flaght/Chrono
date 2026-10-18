"""EMA统一正式回测主链的分阶段验证。

N5使用内存Bar验证Runner统一积木链；N6/N7分别加入真实CTP Bar/Tick文件。
全部使用SimulationExecutionClient和NautilusSimExecutionBackend，不连接真实交易端。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from examples.single_ema.ema_ctp_backtest import run_case
from tests.run_unified_historical_runtime import test3_real_nautilus_pipeline


def test5_unified_backtest_runtime() -> None:
    """N5：内存Bar通过统一Runner主链进入原生撮合引擎。"""

    test3_real_nautilus_pipeline()
    print("N5通过：统一Runner/Planner/Risk/Simulation Backend正式回测")


def test6_ctp_bar_backtest() -> None:
    """N6：真实rb2704 Feather Bar进入正式撮合链路。"""

    run_case("bar")
    print("N6通过：CTP Feather Bar正式回测")


def test7_ctp_tick_backtest() -> None:
    """N7：真实rb2609 Tick经内部聚合Bar后进入正式撮合链路。"""

    run_case("tick")
    print("N7通过：CTP Tick内部聚合正式回测")


STAGES = {
    5: test5_unified_backtest_runtime,
    6: test6_ctp_bar_backtest,
    7: test7_ctp_tick_backtest,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="正式策略运行时分阶段验证")
    parser.add_argument(
        "--stage",
        choices=(*(str(stage) for stage in STAGES), "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "all":
        # Bomber/Nautilus的Rust日志器是进程级单例。一个测试函数释放
        # BacktestEngine后，在同一Python进程创建第二个引擎会因重复初始化
        # 全局logger而直接panic，无法由Python异常捕获。因此all模式只负责编排，
        # 每个阶段放入独立子进程；单阶段模式仍便于直接调试。
        script = str(Path(__file__).resolve())
        for stage in STAGES:
            print(f"\n=== 启动独立进程验证 N{stage} ===", flush=True)
            subprocess.run(
                [sys.executable, script, "--stage", str(stage)],
                check=True,
            )
        print("Formal strategy tests OK")
        return

    stage = int(args.stage)
    print(f"\n--- N{stage} ---")
    STAGES[stage]()
    print(f"N{stage} formal strategy test OK")


if __name__ == "__main__":
    main()
