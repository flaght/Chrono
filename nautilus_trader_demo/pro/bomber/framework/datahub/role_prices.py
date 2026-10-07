"""DataHub 因果角色复权价；只处理参考数据，不接管行情或下单。

每日角色来自上一交易日的日终表。换约优先使用来源交易日旧、新真实合约
的同日收盘价；若旧合约已到期，显式配置后才允许回退至最近一个共同交易日。
已过去的研究价不回写。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping


ROLES = ("main", "secondary", "near", "far")


class ResearchDataUnavailable(LookupError):
    """角色、价格或可用时间不足；当前决策必须跳过。"""


@dataclass(frozen=True)
class ObservedClose:
    instrument: str
    trading_day: date
    ts_event: int
    close: Decimal

    def __post_init__(self) -> None:
        if not self.instrument or self.ts_event < 0 or not self.close.is_finite() or self.close <= 0:
            raise ValueError("真实合约价格记录无效")


@dataclass(frozen=True)
class RoleAssignment:
    """某交易日生效的角色表；source_day是日终表和换约价的来源交易日。"""

    trading_day: date
    source_day: date
    effective_ns: int
    available_ns: int
    contracts: Mapping[str, str]
    roles: tuple[str, ...] = ROLES

    def __post_init__(self) -> None:
        if self.source_day >= self.trading_day or self.effective_ns < 0 or self.available_ns < 0:
            raise ValueError("角色表必须来自此前交易日，时间不能为负")
        if (not self.roles or len(set(self.roles)) != len(self.roles)
                or set(self.contracts) != set(self.roles)
                or any(not value for value in self.contracts.values())):
            raise ValueError(f"角色表必须包含这些角色的真实合约: {self.roles}")
        object.__setattr__(self, "contracts", MappingProxyType(dict(self.contracts)))


@dataclass(frozen=True)
class RolePrice:
    role: str
    instrument: str
    trading_day: date
    source_day: date
    source_ns: int
    raw_close: Decimal
    single_factor: Decimal
    cumulative_factor: Decimal
    adjusted_close: Decimal


class RolePriceStore:
    """角色复权价 as-of 查询；不产生可成交 Bar 和 Instrument。

    因子采用从样本起点向前累计的固定锚点。若外部pcr_cumfactor的锚点不同，
    应比较相邻日因子比值，而不是比较累计因子的绝对值。
    """

    def __init__(
        self,
        assignments: tuple[RoleAssignment, ...],
        closes: tuple[ObservedClose, ...],
        *,
        missing_roll_policy: str = "raise",
        max_anchor_lookback_trading_days: int = 1,
    ) -> None:
        if missing_roll_policy not in {"raise", "skip", "previous_common"}:
            raise ValueError("missing_roll_policy必须为raise、skip或previous_common")
        if max_anchor_lookback_trading_days < 1:
            raise ValueError("换约锚点回看交易日数必须大于零")
        if not assignments:
            raise ValueError("至少需要一条角色记录")
        ordered = tuple(sorted(assignments, key=lambda row: row.effective_ns))
        self.roles = ordered[0].roles
        if any(row.roles != self.roles for row in ordered):
            raise ValueError("同一角色研究序列的角色集合必须一致")
        if len({row.trading_day for row in ordered}) != len(ordered):
            raise ValueError("角色表交易日重复")
        if any(b.effective_ns <= a.effective_ns or b.trading_day <= a.trading_day
               for a, b in zip(ordered, ordered[1:])):
            raise ValueError("角色生效时间和交易日必须递增")
        self._assignments = ordered
        self._effective_times = tuple(row.effective_ns for row in ordered)

        grouped: dict[str, list[ObservedClose]] = {}
        daily_last: dict[tuple[date, str], ObservedClose] = {}
        for close in closes:
            grouped.setdefault(close.instrument, []).append(close)
            key = (close.trading_day, close.instrument)
            if key not in daily_last or close.ts_event > daily_last[key].ts_event:
                daily_last[key] = close
        observed_days = tuple(sorted({day for day, _ in daily_last}))
        self._closes = {key: tuple(sorted(items, key=lambda row: row.ts_event))
                        for key, items in grouped.items()}
        self._close_times = {key: tuple(row.ts_event for row in items)
                             for key, items in self._closes.items()}

        factors: list[Mapping[str, tuple[Decimal, Decimal] | None]] = []
        gaps: list[tuple[date, str, str]] = []
        anchors: list[tuple[date, str, date, date]] = []
        previous: RoleAssignment | None = None
        for assignment in ordered:
            by_role: dict[str, tuple[Decimal, Decimal] | None] = {}
            for role in self.roles:
                single = Decimal(1)
                previous_factor = None if previous is None else factors[-1][role]
                cumulative = (Decimal(1) if previous is None else
                              None if previous_factor is None else previous_factor[1])
                if previous is not None and previous.contracts[role] != assignment.contracts[role]:
                    anchor_day = assignment.source_day
                    old_key = (anchor_day, previous.contracts[role])
                    new_key = (anchor_day, assignment.contracts[role])
                    if (missing_roll_policy == "previous_common"
                            and (old_key not in daily_last or new_key not in daily_last)):
                        # 只回看有真实行情的前N个交易日，并要求旧、新合约
                        # 在同一天都有收盘价；禁止拼接两个不同日期的价格。
                        before = observed_days[:bisect_left(observed_days, assignment.source_day)]
                        for candidate in reversed(before[-max_anchor_lookback_trading_days:]):
                            candidate_old = (candidate, previous.contracts[role])
                            candidate_new = (candidate, assignment.contracts[role])
                            if candidate_old in daily_last and candidate_new in daily_last:
                                anchor_day = candidate
                                old_key, new_key = candidate_old, candidate_new
                                break
                    if old_key not in daily_last or new_key not in daily_last:
                        reason = (
                            f"{assignment.trading_day}/{role}换约缺少来源日"
                            f"{assignment.source_day}旧、新真实合约收盘价"
                        )
                        if missing_roll_policy == "previous_common":
                            reason += f"，且前{max_anchor_lookback_trading_days}个交易日无共同收盘价"
                        if missing_roll_policy == "raise":
                            raise ResearchDataUnavailable(reason)
                        gaps.append((assignment.trading_day, role, reason))
                        cumulative = None
                        by_role[role] = None
                        continue
                    if (daily_last[old_key].ts_event >= assignment.effective_ns
                            or daily_last[new_key].ts_event >= assignment.effective_ns):
                        raise ResearchDataUnavailable(
                            f"{assignment.trading_day}/{role}换约引用尚未发生的旧、新收盘价",
                        )
                    single = daily_last[old_key].close / daily_last[new_key].close
                    if anchor_day != assignment.source_day and cumulative is not None:
                        anchors.append((assignment.trading_day, role,
                                        assignment.source_day, anchor_day))
                    if cumulative is not None:
                        cumulative *= single
                by_role[role] = None if cumulative is None else (single, cumulative)
            factors.append(MappingProxyType(by_role))
            previous = assignment
        self._factors = tuple(factors)
        self._factor_gaps = tuple(gaps)
        self._factor_anchors = tuple(anchors)

    @property
    def factor_gaps(self) -> tuple[tuple[date, str, str], ...]:
        """因缺少真实收盘价而未计算的换约；后续同角色累计序列也不可用。"""
        return self._factor_gaps

    @property
    def factor_anchors(self) -> tuple[tuple[date, str, date, date], ...]:
        """回退锚点审计记录：（生效日、角色、来源日、实际共同收盘日）。"""
        return self._factor_anchors

    def assignment_at(self, as_of_ns: int) -> RoleAssignment:
        index = bisect_right(self._effective_times, as_of_ns) - 1
        if index < 0:
            raise ResearchDataUnavailable("决策时刻尚无生效角色")
        assignment = self._assignments[index]
        if assignment.available_ns > as_of_ns:
            raise ResearchDataUnavailable("最新生效角色尚不可用，不能沿用旧角色")
        return assignment

    def factor_at(self, as_of_ns: int, role: str) -> tuple[Decimal, Decimal]:
        """返回指定角色的（本次换约因子，样本起点以来累计因子）。"""
        if role not in self.roles:
            raise KeyError(role)
        self.assignment_at(as_of_ns)
        index = bisect_right(self._effective_times, as_of_ns) - 1
        factor = self._factors[index][role]
        if factor is None:
            raise ResearchDataUnavailable(
                f"{self._assignments[index].trading_day}/{role}复权因子不可用：此前换约缺价",
            )
        return factor

    def snapshot(self, as_of_ns: int, *, max_source_age_ns: int | None = None) -> Mapping[str, RolePrice]:
        if max_source_age_ns is not None and max_source_age_ns < 0:
            raise ValueError("最大价格年龄不能为负")
        assignment = self.assignment_at(as_of_ns)
        index = bisect_right(self._effective_times, as_of_ns) - 1
        result: dict[str, RolePrice] = {}
        for role in self.roles:
            instrument = assignment.contracts[role]
            times = self._close_times.get(instrument, ())
            position = bisect_right(times, as_of_ns) - 1
            if position < 0:
                raise ResearchDataUnavailable(f"{role}/{instrument}没有当时可见的真实价格")
            observed = self._closes[instrument][position]
            if observed.trading_day > assignment.trading_day:
                raise ResearchDataUnavailable("真实价格的交易日晚于当前角色")
            if max_source_age_ns is not None and as_of_ns - observed.ts_event > max_source_age_ns:
                raise ResearchDataUnavailable(f"{role}/{instrument}真实价格过旧")
            factor = self._factors[index][role]
            if factor is None:
                raise ResearchDataUnavailable(
                    f"{assignment.trading_day}/{role}复权因子不可用：此前换约缺价",
                )
            single, cumulative = factor
            result[role] = RolePrice(
                role, instrument, assignment.trading_day, assignment.source_day,
                observed.ts_event, observed.close, single, cumulative,
                observed.close * cumulative,
            )
        return MappingProxyType(result)
