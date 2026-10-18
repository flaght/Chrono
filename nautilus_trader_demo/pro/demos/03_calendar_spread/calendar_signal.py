"""跨期业务规则：按角色选双腿，并用历史价差窗口生成方向。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date
from math import isfinite, sqrt
import re
from typing import Mapping

ROLE_COLUMNS = {"main": "main", "secondary": "second", "far": "far"}
ROLE_ORDER = tuple(ROLE_COLUMNS)


def available_pair(symbols: Mapping[str, str], leg1_role: str, leg2_role: str,
                   present: set[str]) -> tuple[str, str] | None:
    """仅在请求的角色范围内替换缺行情双腿，不选择 recent。"""
    leg1_index = ROLE_ORDER.index(leg1_role)
    leg2_index = ROLE_ORDER.index(leg2_role)
    if leg1_index >= leg2_index:
        raise ValueError("leg1-role 必须位于 leg2-role 之前：main、secondary、far")
    leg1 = next((symbols[role] for role in ROLE_ORDER[leg1_index:leg2_index]
                 if role in symbols and symbols[role] in present), None)
    leg2 = next((symbols[role] for role in reversed(ROLE_ORDER[leg1_index + 1:leg2_index + 1])
                if role in symbols and symbols[role] in present and symbols[role] != leg1), None)
    return (leg1, leg2) if leg1 is not None and leg2 is not None else None


def select_roles(roles, days: tuple[date, ...], product: str, *,
                 available: Mapping[date, set[str]],
                 leg1_role: str | None = None, leg2_role: str | None = None,
                 missing_role_policy: str = "raise",
                 near_role: str | None = None, far_role: str | None = None):
    """使用严格早于交易日的最新角色记录；文件读取由 dataprep 完成。"""
    import pandas as pd

    # 旧参数仅为第一、第二腿的别名，不代表额外选择第三个合约。
    if (leg1_role is not None and near_role is not None and leg1_role != near_role
            or leg2_role is not None and far_role is not None and leg2_role != far_role):
        raise ValueError("新旧双腿角色参数冲突")
    leg1_role = leg1_role or near_role or "secondary"
    leg2_role = leg2_role or far_role or "far"
    product = product.strip().upper()
    if missing_role_policy not in {"raise", "next-available"}:
        raise ValueError("missing_role_policy 必须为 raise 或 next-available")
    if ROLE_ORDER.index(leg1_role) >= ROLE_ORDER.index(leg2_role):
        raise ValueError("leg1-role 必须位于 leg2-role 之前：main、secondary、far")
    required = {"trade_date", "code", ROLE_COLUMNS[leg1_role], ROLE_COLUMNS[leg2_role]}
    if required - set(roles.columns):
        raise ValueError(f"角色表缺少列: {sorted(required - set(roles.columns))}")
    rows = roles.loc[roles["code"].astype(str).str.strip().str.upper() == product].copy()
    if rows.empty:
        raise ValueError(f"角色表没有 {product}")
    rows["source_day"] = rows["trade_date"].map(lambda value: pd.Timestamp(value).date())
    rows = rows.sort_values("source_day").drop_duplicates("source_day", keep="last")
    chosen, fallbacks = [], []
    for day in days:
        prior = rows.loc[rows["source_day"] < day]
        if prior.empty:
            raise ValueError(f"{day} 没有上一交易日 {product} 角色记录")
        row = prior.iloc[-1]
        if "available_ns" in row and pd.notna(row["available_ns"]):
            from bomber.framework.dataprep.session import fail
            fail("UNSUPPORTED_CAPABILITY", "跨期选约的延迟发布角色需按可用时间门控；当前场景仅支持日初可用资料")
        symbols = {role: str(row[column]).strip().lower()
                   for role, column in ROLE_COLUMNS.items() if column in rows.columns}
        leg1, leg2 = symbols[leg1_role], symbols[leg2_role]
        pattern = rf"{re.escape(product)}\d+"
        if any(re.fullmatch(pattern, item, re.IGNORECASE) is None for item in (leg1, leg2)):
            raise ValueError(f"{day} 的 {product} 两期限角色无效: {leg1}, {leg2}")
        if leg1 == leg2:
            raise ValueError(f"{day} 的两个期限角色指向同一合约: {leg1}")
        present = available[day]
        if leg1 not in present or leg2 not in present:
            if missing_role_policy == "raise":
                missing = [item for item in (leg1, leg2) if item not in present]
                raise FileNotFoundError(f"{day} 角色表合约缺少真实 Bar: {missing}")
            original = (leg1, leg2)
            replacement = available_pair(symbols, leg1_role, leg2_role, present)
            if replacement is None:
                raise FileNotFoundError(f"{day} {product} 角色表期限 {original} 缺 Bar，且无法找到两个有 Bar 的替代期限")
            leg1, leg2 = replacement
            fallbacks.append(f"{day}: {original[0]}/{original[1]} -> {leg1}/{leg2}")
        chosen.append((day, leg1, leg2))
    return tuple(chosen), tuple(fallbacks)


@dataclass(frozen=True)
class CalendarSignal:
    spread: float
    z_score: float
    direction: int  # 1 多第一腿空第二腿；-1 相反；0 空仓。


class CalendarSpreadSignal:
    """当前价差只比较此前 lookback 根同步价差；换约后重新预热。"""

    def __init__(self, lookback: int, entry_z: float, exit_z: float) -> None:
        if lookback < 2 or not all(isfinite(value) for value in (entry_z, exit_z)) or not 0 <= exit_z < entry_z:
            raise ValueError("窗口至少2根，阈值须满足 0 <= exit_z < entry_z 且有限")
        self.lookback, self.entry_z, self.exit_z = lookback, entry_z, exit_z
        self.history: deque[float] = deque(maxlen=lookback)

    def reset(self) -> None:
        self.history.clear()

    def update(self, spread: float, previous_direction: int) -> CalendarSignal | None:
        if not isfinite(spread) or previous_direction not in {-1, 0, 1}:
            raise ValueError("价差须有限，方向须为 -1、0 或 1")
        result = None
        if len(self.history) == self.lookback:
            mean = sum(self.history) / self.lookback
            variance = sum((value - mean) ** 2 for value in self.history) / self.lookback
            z = (spread - mean) / sqrt(variance) if variance > 0 else 0.0
            direction = previous_direction
            if abs(z) <= self.exit_z:
                direction = 0
            elif z <= -self.entry_z:
                direction = 1
            elif z >= self.entry_z:
                direction = -1
            result = CalendarSignal(spread, z, direction)
        # 必须在计算后写入当前值，避免当前价差进入自己的统计基准。
        self.history.append(spread)
        return result
