"""在角色表顺序内挑选当日确有真实 Bar 的两个期限。"""

from __future__ import annotations

from typing import Mapping

ROLE_COLUMNS = {"main": "main", "secondary": "second", "far": "far"}
ROLE_ORDER = tuple(ROLE_COLUMNS)


def available_pair(symbols: Mapping[str, str], near_role: str, far_role: str,
                   present: set[str]) -> tuple[str, str] | None:
    near_index = ROLE_ORDER.index(near_role)
    far_index = ROLE_ORDER.index(far_role)
    if near_index >= far_index:
        raise ValueError("near-role 必须位于 far-role 之前：main、secondary、far")
    near = next((symbols[role] for role in ROLE_ORDER[near_index:far_index]
                 if role in symbols and symbols[role] in present), None)
    far = next((symbols[role] for role in reversed(ROLE_ORDER[near_index + 1:far_index + 1])
                if role in symbols and symbols[role] in present and symbols[role] != near), None)
    return (near, far) if near is not None and far is not None else None
