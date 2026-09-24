"""过期近月缺 Bar 时只选择所请求角色范围内可交易的两个期限。"""

from demos.calendar_spread.selection import available_pair


def test_secondary_far_pair_ignores_recent() -> None:
    symbols = {"main": "i2605", "secondary": "i2609", "far": "i2701"}
    present = {"i2605", "i2609", "i2701"}
    assert available_pair(symbols, "secondary", "far", present) == ("i2609", "i2701")
    assert available_pair(symbols, "main", "secondary", present) == ("i2605", "i2609")


def test_no_two_contracts() -> None:
    symbols = {"main": "i2605", "secondary": "i2609", "far": "i2701"}
    assert available_pair(symbols, "secondary", "far", {"i2605"}) is None


if __name__ == "__main__":
    test_secondary_far_pair_ignores_recent()
    test_no_two_contracts()
    print("calendar selection: OK")
