"""第五类分阶段验证：先检查因果角色、收益率窗口与下一根执行。"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal

from datahub.sector_roles import SectorDataUnavailable, SectorRoleAssignment, SectorRoleStore
from examples.black_sector.sector_signal import BlackSectorSignal


def _store() -> SectorRoleStore:
    return SectorRoleStore((
        SectorRoleAssignment(date(2026, 1, 6), date(2026, 1, 5), 100, 110,
                             {"JM": {"secondary": "jm2605"},
                              "I": {"secondary": "i2605"},
                              "RB": {"secondary": "rb2605", "main": "rb2605"}},
                             {key: {"secondary": Decimal(1)} for key in ("JM", "I", "RB")}),
        SectorRoleAssignment(date(2026, 1, 7), date(2026, 1, 6), 200, 210,
                             {"JM": {"secondary": "jm2609"},
                              "I": {"secondary": "i2609"},
                              "RB": {"secondary": "rb2605", "main": "rb2610"}},
                             {"JM": {"secondary": Decimal("0.9")},
                              "I": {"secondary": Decimal("1.1")},
                              "RB": {"secondary": Decimal(1)}}),
    ))


def test1_roles() -> None:
    store = _store()
    for ns in (99, 100, 205):
        try:
            store.snapshot(ns)
        except SectorDataUnavailable:
            pass
        else:
            raise AssertionError(f"{ns}不应看到未发布的角色")
    assert store.snapshot(110).instrument("RB", "main") == "rb2605"
    assert store.snapshot(210).instrument("RB", "main") == "rb2610"
    assert store.snapshot(210).factor("JM", "secondary") == Decimal("0.9")
    generic = SectorRoleStore((SectorRoleAssignment(
        date(2026, 1, 6), date(2026, 1, 5), 1, 1,
        {"AU": {"active": "au2606"}}, {"AU": {"active": Decimal("1.02")}},
    ),))
    assert generic.snapshot(1).instrument("AU", "active") == "au2606"
    print("V1通过：DataHub不写死品种/角色，合约与因子按生效/发布时间因果查询")


def test2_signal() -> None:
    signal = BlackSectorSignal(3, 2)
    assert signal.update(1, {"JM": Decimal(100), "I": Decimal(100), "RB": Decimal(100)}) is None
    for ts in (2, 3, 4):
        result = signal.update(ts, {"JM": Decimal(100), "I": Decimal(100), "RB": Decimal(100 + ts)})
        assert result is None
    result = signal.update(5, {"JM": Decimal(100), "I": Decimal(100), "RB": Decimal(110)})
    assert result is not None and result.direction == 1 and result.sector_ma == 0
    result = signal.update(6, {"JM": Decimal(100), "I": Decimal(100), "RB": Decimal(90)})
    assert result is not None and result.direction == -1
    other = BlackSectorSignal(1, 1, leader_products=("CU", "AL"), comparison_product="ZN")
    assert other.update(1, {"CU": Decimal(100), "AL": Decimal(100), "ZN": Decimal(100)}) is None
    result = other.update(2, {"CU": Decimal(100), "AL": Decimal(100), "ZN": Decimal(101)})
    assert result is not None and result.direction == 1
    print("V2通过：首帧预热、双层窗口无前视，策略配置可指定领头/比较品种")


def test3_strategy() -> None:
    # 只有这一阶段需要Bomber/Nautilus模型；前两阶段可单独在轻环境执行。
    from market.basic.base import Bar, BarType, InstrumentId, Price, Quantity
    from examples.black_sector.sector_strategy import BlackSectorConfig, BlackSectorTargetStrategy

    class Context:
        def __init__(self) -> None:
            self.intents = []

        def submit(self, intent) -> None:
            self.intents.append(intent)

        def position(self, target_key: str) -> Decimal:
            return Decimal(0)

    def bar(symbol: str, ts: int, close: int) -> Bar:
        instrument = InstrumentId.from_str(f"{symbol}.DCE" if symbol.startswith(("jm", "i")) else f"{symbol}.SHFE")
        bar_type = BarType.from_str(f"{instrument}-1-MINUTE-LAST-EXTERNAL")
        price = Price.from_str(str(close))
        return Bar(bar_type, price, price, price, price, Quantity.from_str("1"), ts, ts)

    # 缩短窗口只为验证机制；正式默认仍是30/15。
    store = SectorRoleStore((SectorRoleAssignment(
        date(2026, 1, 6), date(2026, 1, 5), 1, 1,
        {"JM": {"secondary": "jm2605"},
         "I": {"secondary": "i2605"},
         "RB": {"secondary": "rb2605", "main": "rb2610"}},
        {key: {"secondary": Decimal(1)} for key in ("JM", "I", "RB")}),))
    strategy = BlackSectorTargetStrategy(
        "sector-test", store,
        BlackSectorConfig(return_period=1, sector_period=1, submission_delay_bars=1),
    )
    context = Context()
    strategy._bind(context)
    strategy._start()
    for ts, rb in ((10, 100), (20, 110)):
        for symbol, close in (("jm2605", 100), ("i2605", 100), ("rb2605", rb)):
            strategy._handle_event(symbol, bar(symbol, ts, close))
    assert not context.intents
    # 同时间戳的主力Bar不能触发信号的下一根执行。
    strategy._handle_event("rb2610", bar("rb2610", 20, 100))
    assert not context.intents
    strategy._handle_event("rb2610", bar("rb2610", 30, 100))
    assert len(context.intents) == 1
    assert context.intents[0].targets["rb_main"] == Decimal(1)
    assert context.intents[0].metadata["signal_ns"] == 20
    assert context.intents[0].metadata["target_submission_ns"] == 30
    strategy._stop()
    print("V3通过：真实Bar三品种同步、只提交RB主力目标，等待目标合约下一根Bar")


STAGES = {"roles": test1_roles, "signal": test2_signal, "strategy": test3_strategy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    args = parser.parse_args()
    for name, function in STAGES.items():
        if args.stage in (name, "all"):
            function()


if __name__ == "__main__":
    main()
