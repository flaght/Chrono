"""第四类M1/M2：四角色因果复权与example10实际文件探针。

默认纯离线夹具；--real读取用户提供的三张参考表和真实RB合约Bar。
此阶段只验研究数据，不宣称已完成策略撮合或正式换月。
"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path

from datahub import (
    MinimalDataHub,
    ObservedClose,
    ResearchDataUnavailable,
    RoleAssignment,
    RolePriceStore,
)


def _factor_alignment(calculated: Decimal, recorded: Decimal | None) -> str:
    if recorded is None or not recorded.is_finite() or recorded <= 0:
        return "missing"
    tolerance = Decimal("0.000001")
    if abs(calculated / recorded - 1) <= tolerance:
        return "direct"
    if abs(calculated * recorded - 1) <= tolerance:
        return "inverse"
    return "mismatch"


def _assignment(day: int, source: int, effective: int, available: int, **contracts: str) -> RoleAssignment:
    source_day = date(2026, 6, 30) if source == 0 else date(2026, 7, source)
    return RoleAssignment(date(2026, 7, day), source_day, effective, available, contracts)


def test_m1_m2() -> None:
    closes = (
        ObservedClose("rb_a", date(2026, 7, 1), 100, Decimal(100)),
        ObservedClose("rb_b", date(2026, 7, 1), 100, Decimal(110)),
        ObservedClose("rb_c", date(2026, 7, 1), 100, Decimal(90)),
        ObservedClose("rb_d", date(2026, 7, 1), 100, Decimal(120)),
        ObservedClose("rb_a", date(2026, 7, 2), 200, Decimal(101)),
        ObservedClose("rb_b", date(2026, 7, 2), 200, Decimal(111)),
        ObservedClose("rb_c", date(2026, 7, 2), 200, Decimal(92)),
        ObservedClose("rb_d", date(2026, 7, 2), 200, Decimal(121)),
    )
    store = RolePriceStore((
        _assignment(1, 0, 100, 100, main="rb_a", secondary="rb_b", near="rb_c", far="rb_d"),
        _assignment(2, 1, 200, 200, main="rb_b", secondary="rb_a", near="rb_c", far="rb_d"),
        _assignment(3, 2, 300, 350, main="rb_a", secondary="rb_b", near="rb_c", far="rb_d"),
    ), closes)
    hub = MinimalDataHub(store)
    assert hub.snapshot(200).contracts["main"] == "rb_b"
    assert hub.snapshot(200).prices["main"].instrument == "rb_b"
    try:
        hub.snapshot(200).contracts["main"] = "rb_a"
    except TypeError:
        pass
    else:
        raise AssertionError("研究数据快照不能被策略改写")
    print("M0通过：DataHub独立提供不可变的角色/研究价as-of快照")
    assert store.factor_at(200, "main")[0] == Decimal(100) / Decimal(110)
    assert store.factor_at(200, "secondary")[0] == Decimal(110) / Decimal(100)
    assert store.factor_at(200, "near")[0] == 1
    assert _factor_alignment(Decimal(100) / Decimal(110), Decimal(100) / Decimal(110)) == "direct"
    assert _factor_alignment(Decimal(100) / Decimal(110), Decimal(110) / Decimal(100)) == "inverse"
    assert store.snapshot(200)["main"].adjusted_close == Decimal(111) * Decimal(100) / Decimal(110)
    assert store.snapshot(199)["main"].instrument == "rb_a"
    try:
        store.assignment_at(320)
    except ResearchDataUnavailable:
        pass
    else:
        raise AssertionError("新角色尚未发布时不得偷看或退回旧角色")
    try:
        store.snapshot(200, max_source_age_ns=0)["main"]
    except ResearchDataUnavailable:
        raise AssertionError("当时的新鲜价格不应被拒绝")
    try:
        store.snapshot(201, max_source_age_ns=0)
    except ResearchDataUnavailable:
        pass
    else:
        raise AssertionError("陈旧价格没有被拒绝")
    delayed = tuple(
        ObservedClose(item.instrument, item.trading_day,
                      250 if item.trading_day == date(2026, 7, 1) else item.ts_event,
                      item.close)
        for item in closes
    )
    try:
        RolePriceStore((
            _assignment(1, 0, 100, 100, main="rb_a", secondary="rb_b", near="rb_c", far="rb_d"),
            _assignment(2, 1, 200, 200, main="rb_b", secondary="rb_a", near="rb_c", far="rb_d"),
        ), delayed)
    except ResearchDataUnavailable:
        pass
    else:
        raise AssertionError("换约不能使用生效时间之后才发生的收盘价")
    print("M1/M2通过：四角色独立复权、因果查询、版本不可用与价格陈旧检测正常")


def test_missing_roll_factor_can_be_skipped_for_probe() -> None:
    """缺旧合约来源日收盘价时只跳过受影响决策，不伪造复权因子。"""
    first = _assignment(16, 15, 100, 100,
                        main="rb_main", secondary="rb_second",
                        near="rb2601", far="rb_far")
    second = _assignment(19, 16, 200, 200,
                         main="rb_main", secondary="rb_second",
                         near="rb2602", far="rb_far")
    third = _assignment(20, 19, 300, 300,
                        main="rb_main", secondary="rb_second",
                        near="rb2602", far="rb_far")
    closes = (
        ObservedClose("rb2601", date(2026, 7, 15), 90, Decimal(3100)),
        ObservedClose("rb2602", date(2026, 7, 16), 150, Decimal(3150)),
        ObservedClose("rb_main", date(2026, 7, 16), 150, Decimal(3200)),
        ObservedClose("rb_second", date(2026, 7, 16), 150, Decimal(3210)),
        ObservedClose("rb_far", date(2026, 7, 16), 150, Decimal(3220)),
        ObservedClose("rb2602", date(2026, 7, 19), 250, Decimal(3160)),
    )
    try:
        RolePriceStore((first, second, third), closes)
    except ResearchDataUnavailable:
        pass
    else:
        raise AssertionError("正式严格模式不能忽略缺失的旧合约收盘价")
    store = RolePriceStore((first, second, third), closes,
                           missing_roll_policy="skip")
    assert len(store.factor_gaps) == 1
    assert store.factor_gaps[0][:2] == (date(2026, 7, 19), "near")
    assert store.factor_at(200, "main") == (Decimal(1), Decimal(1))
    for timestamp in (200, 300):
        try:
            store.factor_at(timestamp, "near")
        except ResearchDataUnavailable:
            pass
        else:
            raise AssertionError("缺口后的near累计因子不能假设为1")
    try:
        store.snapshot(200)
    except ResearchDataUnavailable:
        pass
    else:
        raise AssertionError("四角色不完整时不能产生策略研究快照")
    from examples.role_cross.local_input import _file_trading_day

    assert _file_trading_day("rb2602_20260116.feather") == date(2026, 1, 16)
    print("M2缺口通过：严格模式报错，探针模式记录缺口并跳过不可信决策；夜盘按文件交易日归属")


def test_previous_common_roll_anchor() -> None:
    """旧合约到期后，前一共同交易日锚定必须同日、因果且有界。"""
    old_day = date(2026, 1, 15)
    source_day = date(2026, 1, 16)
    effective_day = date(2026, 1, 19)
    first = RoleAssignment(source_day, old_day, 200, 200, {
        "main": "rb2605", "secondary": "rb2607",
        "near": "rb2601", "far": "rb2610",
    })
    second = RoleAssignment(effective_day, source_day, 300, 300, {
        "main": "rb2605", "secondary": "rb2607",
        "near": "rb2602", "far": "rb2610",
    })
    closes = (
        ObservedClose("rb2601", old_day, 100, Decimal("3165")),
        ObservedClose("rb2602", old_day, 100, Decimal("3135")),
        ObservedClose("rb2602", source_day, 250, Decimal("3150")),
        ObservedClose("rb2602", effective_day, 300, Decimal("3140")),
    )
    store = RolePriceStore((first, second), closes,
                           missing_roll_policy="previous_common")
    expected = Decimal("3165") / Decimal("3135")
    assert store.factor_at(300, "near") == (expected, expected)
    assert store.factor_anchors == ((effective_day, "near", source_day, old_day),)
    assert not store.factor_gaps
    # 前一交易日不是共同收盘日时，不得跨过它使用更早的旧价。
    too_old = (
        ObservedClose("rb2601", date(2026, 1, 14), 50, Decimal("3160")),
        ObservedClose("rb2602", date(2026, 1, 14), 50, Decimal("3130")),
        ObservedClose("rb2602", old_day, 100, Decimal("3135")),
        closes[2],
    )
    bounded = RolePriceStore((first, second), too_old,
                             missing_roll_policy="previous_common")
    assert len(bounded.factor_gaps) == 1
    assert not bounded.factor_anchors
    try:
        bounded.factor_at(300, "near")
    except ResearchDataUnavailable:
        pass
    else:
        raise AssertionError("不能越过缺价交易日使用更早的换约锚点")
    # 即使旧、新同日收盘齐备，也不能引用换约生效之后才到达的价格。
    late = (ObservedClose("rb2601", old_day, 350, Decimal("3165")),) + closes[1:]
    try:
        RolePriceStore((first, second), late,
                       missing_roll_policy="previous_common")
    except ResearchDataUnavailable:
        pass
    else:
        raise AssertionError("共同收盘价不能晚于换约生效时刻")
    print("M2锚点通过：1月19日near使用1月15日同日收盘价，回看有界且无前视")


def probe_real_files(args: argparse.Namespace) -> None:
    from examples.role_cross.local_input import load_rb_role_research

    result = load_rb_role_research(
        bars_dir=args.bars_dir,
        contract_struct_path=args.contract_struct,
        factors_path=args.factors,
        fut_basic_path=args.fut_basic,
    )
    switches = tuple(row for row in result.audit
                     if row.calculated_single is not None and row.calculated_single != 1)
    hub = MinimalDataHub(result.store)
    print(f"M2真实文件探针：bars={result.bar_count} contracts={result.contract_count} "
          f"days={len(result.audit)} main_switches={len(switches)}")
    for day, role, reason in result.store.factor_gaps:
        print(f"  复权缺口：day={day} role={role} reason={reason}；此后该角色快照不可用于信号")
    for day, role, source_day, anchor_day in result.store.factor_anchors:
        print(f"  换约锚点：day={day} role={role} source={source_day} anchor={anchor_day}")
    complete_days = 0
    unavailable_days = 0
    for _, end_ns in result.day_end_ns:
        try:
            hub.snapshot(end_ns)
        except ResearchDataUnavailable:
            unavailable_days += 1
        else:
            complete_days += 1
    print(f"  四角色日终快照：complete_days={complete_days} unavailable_days={unavailable_days}")
    for row in switches[:10]:
        source_match = _factor_alignment(row.calculated_single, row.source_day_single)
        effective_match = _factor_alignment(row.calculated_single, row.effective_day_single)
        print(f"  {row.trading_day} main={row.main_instrument} source={row.source_day} "
          f"calculated={row.calculated_single} "
          f"pcr_symbols=({row.source_day_symbol},{row.effective_day_symbol}) "
          f"pcr_source={row.source_day_single}({source_match}) "
              f"pcr_effective={row.effective_day_single}({effective_match}) "
              f"cum_source={row.source_day_cumulative} cum_effective={row.effective_day_cumulative}")
    if not switches:
        print("注意：样本没有主力切换，尚不能验收因子对账与换月。")
    if not all(row.source_day_single is not None or row.effective_day_single is not None
               for row in switches):
        print("注意：部分切换日附近缺少pcr_factor。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="读取Linux服务器上的实际Feather文件")
    parser.add_argument("--bars-dir", type=Path, default=Path(
        "/workspace/worker/kdwk/Chrono/nautilus_trader_demo/test_data/bars/mo"))
    role_root = Path("/workspace/worker/pj/neutron/tests/temp/role")
    parser.add_argument("--contract-struct", "--fut-contract", type=Path,
                        default=role_root / "fut_contract_data.feather")
    parser.add_argument("--factors", type=Path, default=role_root / "fut_adjustment_factors.feather")
    parser.add_argument("--fut-basic", type=Path, default=role_root / "fut_basic.feather")
    args = parser.parse_args()
    test_m1_m2()
    test_missing_roll_factor_can_be_skipped_for_probe()
    test_previous_common_roll_anchor()
    if args.real:
        probe_real_files(args)


if __name__ == "__main__":
    main()
