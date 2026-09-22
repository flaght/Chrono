"""DataHub阶段D0：格式无关Provider、as-of时间闸门和日终交易日规则。"""

from __future__ import annotations

from datetime import date

from datahub import (
    AsOfQuery,
    DataHub,
    DataHubUnavailable,
    FutureDataError,
    InMemoryAsOfProvider,
    PublicationPolicy,
    ReferenceRecord,
)


FRIDAY = date(2026, 9, 18)
MONDAY = date(2026, 9, 21)


def _expect(error_type: type[Exception], function) -> None:
    try:
        function()
    except error_type:
        return
    raise AssertionError(f"应当抛出{error_type.__name__}")


def test_d0_generic_time_gate() -> None:
    records = (
        ReferenceRecord("contract_roles", "RB", "old", 100, 100, 100, FRIDAY,
                        PublicationPolicy.DAY_END),
        ReferenceRecord("contract_roles", "RB", "new", 200, 250, 180, FRIDAY,
                        PublicationPolicy.DAY_END),
        ReferenceRecord("research_price", "RB.main", 3010, 205, 205, 150),
    )
    provider = InMemoryAsOfProvider(records)
    hub = DataHub({"contract_roles": provider, "research_price": provider})

    # 同一日的日终表，即使时间戳已到，也不能盘中读取。
    _expect(DataHubUnavailable, lambda: hub.get(
        "contract_roles", "RB", AsOfQuery(100, trading_day=FRIDAY)))
    # 周一取上周五；从交易日字段判断，不能用自然日减一天。
    monday = AsOfQuery(190, trading_day=MONDAY)
    assert hub.get("contract_roles", "RB", monday)[0].value == "old"
    # 新版本已生效但尚未发布，不可悄悄回退到旧版本。
    _expect(FutureDataError, lambda: hub.get(
        "contract_roles", "RB", AsOfQuery(220, trading_day=MONDAY)))
    assert hub.get("contract_roles", "RB", AsOfQuery(250, trading_day=MONDAY))[0].value == "new"
    assert tuple(row.value for row in hub.get(
        "contract_roles", "RB", AsOfQuery(250, trading_day=MONDAY, window=2))) == ("old", "new")

    _expect(FutureDataError, lambda: AsOfQuery(200, data_time_ns=201))
    _expect(DataHubUnavailable, lambda: hub.get(
        "research_price", "RB.main", AsOfQuery(210, max_source_age_ns=50)))
    assert hub.get("research_price", "RB.main", AsOfQuery(210, max_source_age_ns=60))[0].value == 3010

    # 多字段快照必须完整；缺少任何一个字段时，不返回半成品。
    complete = hub.snapshot((('contract_roles', 'RB'), ('research_price', 'RB.main')),
                            AsOfQuery(250, trading_day=MONDAY))
    assert complete.latest('contract_roles', 'RB').value == 'new'
    try:
        complete.values[('contract_roles', 'RB')] = ()
    except TypeError:
        pass
    else:
        raise AssertionError("快照映射必须不可变")
    _expect(DataHubUnavailable, lambda: hub.snapshot(
        (('contract_roles', 'RB'), ('research_price', 'RB.far')),
        AsOfQuery(250, trading_day=MONDAY)))

    class BadProvider:
        def read(self, dataset, key, query):
            return (ReferenceRecord(dataset, key, "future", 300, 300, 300),)

    _expect(FutureDataError, lambda: DataHub({"bad": BadProvider()}).get(
        "bad", "x", AsOfQuery(250)))
    print("D0通过：Provider可替换、日终交易日隔离、时间可见性、陈旧检测和完整快照正常")


if __name__ == "__main__":
    test_d0_generic_time_gate()
