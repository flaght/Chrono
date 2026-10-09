"""只读检查EMA断点及文件/数据库分钟覆盖；不连接柜台、不保存状态、不发布目标。"""

import argparse
from datetime import datetime
from decimal import Decimal
import json
from types import SimpleNamespace
from threading import RLock
from pathlib import Path

from bomber.framework.trader.persistence import JsonStateStore
from bomber.framework.trader.runtime.trading_sessions import CopperSessions, SHANGHAI
from .checkpoint import EmaCheckpoint
from .continuous import legacy_ema_state, recover_minutes
from .strategy import MainEmaConfig, MainEmaStrategy


def display_ns(stamp):
    # 避免float把分钟末最后1ns四舍五入到下一分钟。
    return datetime.fromtimestamp(stamp // 1_000_000_000, SHANGHAI).isoformat()


def check(args):
    persisted = JsonStateStore(args.state_file).load()
    if persisted is None:
        raise ValueError("状态文件不存在")
    components = persisted.payload.get("component_states", {})
    saved = components.get("ema")
    targets = persisted.payload["targets"]
    if len(targets) != 1:
        raise ValueError("本检查入口限定单EMA策略检查点")
    target = targets[0]
    if saved is not None:
        config = saved["config"]
        instrument = saved["instrument"]
    else:
        if args.resume_report is None:
            raise ValueError("旧检查点没有EMA值，须提供对应已通过的--resume-report")
        report = json.loads(args.resume_report.read_text())
        instrument = report["fixed_instrument"]
        config = {"product": report["product"], "venue": instrument.split(".")[1],
            "fast": report["ema"]["fast"], "slow": report["ema"]["slow"],
            "quantity": report["ema"]["quantity"], "target_key": next(iter(target["targets"]))}
    strategy = MainEmaStrategy(target["strategy_id"], None, MainEmaConfig(
        config["product"], config["venue"], config["fast"], config["slow"],
        Decimal(config["quantity"]), config["target_key"]))
    checkpoint = EmaCheckpoint(strategy, instrument)
    if saved is None:
        saved = legacy_ema_state(args.resume_report, checkpoint, persisted, args.state_file)["ema"]
    checkpoint.restore_state(saved)
    if (target["revision"] != strategy._revision
            or Decimal(target["targets"][config["target_key"]]) != strategy.last_target):
        raise ValueError("EMA目标和版本与统一TargetStore不一致")
    before = datetime.fromisoformat(args.before) if args.before else datetime.now(SHANGHAI)
    if before.tzinfo is None:
        raise ValueError("--before须携带时区，例如2026-10-09T14:00:00+08:00")
    before_ns = int(before.timestamp() * 1_000_000_000)
    calendar = CopperSessions(args.trading_calendar)
    calendar.window(before)  # 检查覆盖范围，即使当前休市也不能使用过期日历。
    missing = list(calendar.missing_minutes(strategy.last_processed_ns, before_ns))
    print(f"状态generation={persisted.generation} 合约={instrument} "
          f"EMA bars={strategy.bars_used} 已发布目标={strategy.last_target}", flush=True)
    print(f"断点={display_ns(strategy.last_processed_ns)} 检查截止={before.astimezone(SHANGHAI).isoformat()}", flush=True)
    print(f"需要补齐完整交易分钟={len(missing)}", flush=True)
    if missing:
        print(f"首根={display_ns(missing[0])} ts_event={missing[0]}", flush=True)
        print(f"末根={display_ns(missing[-1])} ts_event={missing[-1]}", flush=True)
    incomplete = False
    if args.history_config or args.history_file or args.history_minutes:
        if args.bars:
            raise ValueError("--bars旧JSONL入口不能与新历史Provider参数混用")
        from .history import build_history, warm_history
        runner = SimpleNamespace(ema_checkpoint=checkpoint, calendar=calendar,
            history_service=build_history(args), history_minutes=args.history_minutes,
            _submit_lock=RLock(), _clients={"readonly": SimpleNamespace(_submit_lock=RLock())})
        warm_history(runner, before_ns, phase=args.history_stage)
        print(f"历史Provider覆盖：{runner.history_service.last_audit}", flush=True)
        incomplete = bool(runner.history_service.last_audit["missing"])
        if incomplete:
            print("允许缺失模式仅验证按配置继续；历史覆盖仍不完整。", flush=True)
    else:
        recover_minutes(checkpoint, calendar, before_ns, args.bars)
    label = "允许缺失恢复预检完成（覆盖不完整）" if incomplete else "断点/分钟覆盖检查通过"
    print(f"{label}：内存EMA bars={strategy.bars_used}；"
          "未连接CTP、未改写状态、未发布目标。此结果不替代柜台对账和实时衔接验收。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument("--bars", type=Path, help="真实完整分钟JSONL；省略时只检查是否存在缺口")
    parser.add_argument("--before", help="固定检查截止时刻，须携带时区；默认当前北京时间")
    parser.add_argument("--history-db", default="dolphindb")
    parser.add_argument("--history-config", type=Path)
    parser.add_argument("--history-file", type=Path)
    parser.add_argument("--history-minutes", type=int, default=0)
    parser.add_argument("--history-stage", choices=("preopen", "recovery"), default="recovery")
    parser.add_argument("--history-missing-policy", choices=("fail", "allow"), default="fail")
    parser.add_argument("--reference-timeout", type=int, default=15)
    parser.add_argument("--trading-calendar", type=Path,
        default=Path(__file__).with_name("shfe_cu_2026.json"))
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        check(args)
    except Exception as error:
        print(f"恢复预检未通过：{error}", flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
