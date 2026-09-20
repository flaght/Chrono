"""目标保存、组合净额和仓位状态的分阶段验证脚本。

三个组件位于策略与执行客户端之间：

``TargetStore``
    回答“每个策略最新想持有什么”，处理REPLACE/PATCH、版本和过期目标。
``PortfolioCoordinator``
    回答“多个策略汇总后，每个客户端账户最终想持有什么”。
``PositionManager``
    回答“策略归属和账户实际上已经持有什么，还有多少订单在途”。

前三个测试分别隔离验证一个组件；第四个测试手工串联；第五个测试把三者正式接入
Runner。这里不创建订单、不撮合，也没有动态主力解析或真实交易回报。
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from market.basic.base import InstrumentId  # noqa: E402
from strategy import (  # noqa: E402
    AccountTargetKey,
    DataBinding,
    ExecutionRequest,
    ExecutionRoute,
    PortfolioCoordinator,
    PositionManager,
    RuntimeMode,
    StaleRevisionError,
    StrategyTemplate,
    TargetExpiredError,
    TargetPortfolio,
    TargetStore,
    TargetUpdateMode,
    UnifiedStrategyRunner,
)
from market.basic.base import DataType, MarketDataFeed, SubscriptionRequest  # noqa: E402


RB = InstrumentId.from_str("rb2610.SHFE")
IF = InstrumentId.from_str("IF2610.CFFEX")


class _ManualFeed(MarketDataFeed):
    """E4只验证装配所需的最小Feed，不产生行情。"""

    def connect(self) -> None:
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        del request

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        del request


class _StrategyProbe(StrategyTemplate):
    pass


class _RecordingClient:
    """只记录账户净额快照，不创建任何订单。"""

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.requests: list[ExecutionRequest] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def submit_targets(self, request: ExecutionRequest) -> None:
        if not self.started:
            raise RuntimeError("记录客户端尚未启动")
        self.requests.append(request)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id


def test1_target_store() -> None:
    """阶段1：验证单个策略目标的保存、物化和时序保护。

    测试依次覆盖：首次完整目标、PATCH局部修改、REPLACE完整替换、重复revision
    拒绝以及deadline过期拒绝。它验证内存语义，不验证落盘和重启恢复。
    """

    store = TargetStore()

    # revision=1是alpha策略的第一份完整目标：RB多2手，IF空1手。
    first = store.apply(
        TargetPortfolio(
            strategy_id="alpha",
            revision=1,
            ts_event=100,
            targets={"rb": 2, "if": -1},
        ),
    )
    assert dict(first.targets) == {"rb": Decimal(2), "if": Decimal(-1)}

    # PATCH只修改rb；未出现的if必须继续保持-1。
    patched = store.apply(
        TargetPortfolio(
            strategy_id="alpha",
            revision=2,
            ts_event=200,
            targets={"rb": 0},
            update_mode=TargetUpdateMode.PATCH,
        ),
    )
    assert dict(patched.targets) == {"rb": Decimal(0), "if": Decimal(-1)}
    # Store对外返回的是合并后的完整快照，所以物化结果统一标记为REPLACE。
    assert patched.update_mode is TargetUpdateMode.REPLACE

    # 默认模式是REPLACE。新快照只包含rb，因此旧的if目标被移除。
    replaced = store.apply(
        TargetPortfolio(
            strategy_id="alpha",
            revision=3,
            ts_event=300,
            targets={"rb": 1},
        ),
    )
    assert dict(replaced.targets) == {"rb": Decimal(1)}

    # 当前版本已经是3，再收到3属于重复消息，必须拒绝而不能覆盖当前状态。
    try:
        store.apply(
            TargetPortfolio(
                strategy_id="alpha",
                revision=3,
                ts_event=301,
                targets={"rb": 2},
            ),
        )
    except StaleRevisionError:
        pass
    else:
        raise AssertionError("重复 revision 应被拒绝")

    # deadline=150而处理时间now=151，目标已经失效，不应继续送往执行层。
    try:
        store.apply(
            TargetPortfolio(
                strategy_id="expired",
                revision=1,
                ts_event=100,
                targets={"rb": 1},
                deadline_ns=150,
            ),
            now_ns=151,
        )
    except TargetExpiredError:
        pass
    else:
        raise AssertionError("过期目标应被拒绝")

    print("阶段1通过：TargetStore支持REPLACE/PATCH、版本和过期校验")


def test2_portfolio_coordinator() -> None:
    """阶段2：验证多策略目标按真实账户腿净额并保留归属。

    alpha-a和alpha-b在同一个Bomber客户端交易RB，因此可以净额；相同RB位于
    vn.py客户端时属于另一个账户边界，不能与Bomber净额。
    """

    coordinator = PortfolioCoordinator()

    # AccountTargetKey = 执行客户端边界 + 真实InstrumentId。
    bomber_rb = AccountTargetKey("bomber-ctp", RB)
    vnpy_rb = AccountTargetKey("vnpy-ctp", RB)

    # alpha-a希望Bomber账户持有RB +2。
    coordinator.update(
        strategy_id="alpha-a",
        revision=1,
        ts_event=100,
        targets={bomber_rb: 2},
    )
    # alpha-b在Bomber账户贡献-1，同时在vn.py账户贡献-3。
    netted = coordinator.update(
        strategy_id="alpha-b",
        revision=1,
        ts_event=100,
        targets={bomber_rb: -1, vnpy_rb: -3},
    )
    # 同客户端同合约净额：+2 + (-1) = +1。
    assert netted.targets[bomber_rb] == 1
    # 不同客户端不净额，所以vn.py目标仍为-3。
    assert netted.targets[vnpy_rb] == -3
    # 净额后仍保留策略明细，后续才能做策略归属、退出和分策略PnL。
    assert netted.contributions["alpha-a"][bomber_rb] == 2
    assert netted.contributions["alpha-b"][bomber_rb] == -1

    # alpha-a退出后只剩alpha-b，Bomber账户目标由+1变成-1。
    removed = coordinator.remove_strategy("alpha-a", ts_event=200)
    assert removed.targets[bomber_rb] == -1
    coordinator.remove_strategy("alpha-b", ts_event=300)
    # 所有策略退出后仍保留已知账户腿并输出0，明确告诉下游需要清仓。
    flat = coordinator.snapshot(ts_event=300)
    assert flat.targets[bomber_rb] == 0
    assert flat.targets[vnpy_rb] == 0
    print("阶段2通过：PortfolioCoordinator按账户净额并保留策略贡献")


def test3_position_manager() -> None:
    """阶段3：验证策略归属、账户真实仓位和在途数量相互独立。

    ``strategy_position`` 是分策略归属；``account_position`` 是交易端权威仓位；
    ``working_quantity`` 是未完全成交的净委托数量。三者不能混为一个字段。
    """

    positions = PositionManager()

    # 策略alpha-a在逻辑腿rb_main上的归属仓位是+2。
    positions.set_strategy_position("alpha-a", "rb_main", 2, revision=1)
    assert positions.position("alpha-a", "rb_main") == 2
    assert positions.position("unknown", "rb_main") == 0

    # Bomber账户真实持有RB +1，同时还有净买入2手在途。
    positions.set_account_position("bomber-ctp", RB, 1, revision=10)
    positions.set_working_quantity("bomber-ctp", RB, 2)
    assert positions.account_position("bomber-ctp", RB) == 1
    assert positions.working_quantity("bomber-ctp", RB) == 2
    # 有效仓位=真实仓位+在途数量=1+2=3，用于避免重复下单。
    assert positions.effective_position("bomber-ctp", RB) == 3

    # 权威全量快照只包含IF=-1；未出现的旧RB必须归零，不能留下幽灵仓位。
    positions.replace_account_positions("bomber-ctp", {IF: -1})
    assert positions.account_position("bomber-ctp", RB) == 0
    assert positions.account_position("bomber-ctp", IF) == -1

    # alpha-a/rb_main已有revision=1，重复版本不能把归属仓位改成3。
    try:
        positions.set_strategy_position("alpha-a", "rb_main", 3, revision=1)
    except StaleRevisionError:
        pass
    else:
        raise AssertionError("倒序策略仓位版本应被拒绝")

    # 测试中直接清理在途状态；生产环境必须先查询并核对真实活动订单。
    positions.clear_working(client_id="bomber-ctp")
    assert positions.working_quantity("bomber-ctp", RB) == 0
    print("阶段3通过：PositionManager区分策略、账户真实仓位和在途数量")


def test4_combined_flow() -> None:
    """阶段4：串联三个组件，验证“是否还需要下单”的核心公式。

    两个策略的逻辑目标先保存，再用静态映射模拟未来的TargetResolver，随后净额
    成账户目标。最后用账户真实仓位和在途数量计算剩余执行量。

    本测试只计算结果，不会创建订单；撤单、反向目标和部分成交仍需后续
    ExecutionCoordinator处理。
    """

    store = TargetStore()
    coordinator = PortfolioCoordinator()
    positions = PositionManager()
    key = AccountTargetKey("bomber-ctp", RB)

    # 两个策略都交易逻辑腿rb_main：alpha-a=+2，alpha-b=-1。
    for strategy_id, quantity in (("alpha-a", 2), ("alpha-b", -1)):
        target = store.apply(
            TargetPortfolio(
                strategy_id=strategy_id,
                revision=1,
                ts_event=100,
                targets={"rb_main": quantity},
            ),
        )
        # 当前用静态映射代替后续DynamicTargetResolver：rb_main → rb2610.SHFE。
        coordinator.update(
            strategy_id=strategy_id,
            revision=target.revision,
            ts_event=target.ts_event,
            targets={key: target.targets["rb_main"]},
        )

    # 多策略净额后的Bomber/RB账户目标为+1。
    account_target = coordinator.snapshot(ts_event=100).targets[key]

    # 真实仓位仍为0，但已有买入1手在途，所以有效仓位已经达到+1。
    positions.set_account_position("bomber-ctp", RB, 0)
    positions.set_working_quantity("bomber-ctp", RB, 1)
    # 待执行量=账户目标-(真实仓位+在途数量)=1-(0+1)=0。
    remaining = account_target - positions.effective_position("bomber-ctp", RB)
    assert account_target == 1
    assert remaining == 0
    print("阶段4通过：组合目标可与真实仓位及在途数量正确对比")


def test5_runner_integration() -> None:
    """阶段5/E4：验证三个核心组件已进入Runner真实提交链路。

    alpha先贡献RB +2、IF -3，beta再贡献RB -1。同一客户端账户收到的第二份
    快照必须净额为RB +1、IF -3。alpha随后用REPLACE只保留RB=0，被移除的IF
    贡献必须清零，账户最终变为RB -1、IF 0。
    """

    feed = _ManualFeed("manual")
    client = _RecordingClient("recording")
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    runner.add_data_feed("manual", feed)
    runner.add_execution_client(client)

    binding = DataBinding(
        data_key="unused",
        feed_id="manual",
        instrument_id=RB,
        data_type=DataType.TRADE_TICK,
    )
    runner.add_strategy(
        _StrategyProbe("alpha"),
        data_bindings=(binding,),
        execution_routes=(
            ExecutionRoute("rb", "recording", RB),
            ExecutionRoute("if", "recording", IF),
        ),
    )
    runner.add_strategy(
        _StrategyProbe("beta"),
        data_bindings=(binding,),
        execution_routes=(ExecutionRoute("rb", "recording", RB),),
    )

    runner.start()
    try:
        # 未配置的目标键必须在写入TargetStore前失败；随后revision=1仍应合法。
        try:
            runner.submit(TargetPortfolio("alpha", 1, 99, {"unknown": 1}))
        except ValueError:
            pass
        else:
            raise AssertionError("未知target_key必须被拒绝")
        assert runner.target_store.get("alpha") is None

        runner.submit(TargetPortfolio("alpha", 1, 100, {"rb": 2, "if": -3}))
        runner.submit(TargetPortfolio("beta", 1, 101, {"rb": -1}))
        runner.submit(TargetPortfolio("alpha", 2, 102, {"rb": 0}))

        assert dict(client.requests[0].targets) == {RB: Decimal(2), IF: Decimal(-3)}
        assert dict(client.requests[1].targets) == {RB: Decimal(1), IF: Decimal(-3)}
        assert dict(client.requests[2].targets) == {RB: Decimal(-1), IF: Decimal(0)}
        assert dict(client.requests[2].logical_targets) == {
            "rb": Decimal(0),
            "if": Decimal(0),
        }
        assert dict(runner.target_store.get("alpha").targets) == {"rb": Decimal(0)}
        snapshot = runner.portfolio_coordinator.snapshot(ts_event=102)
        assert snapshot.targets[AccountTargetKey("recording", RB)] == Decimal(-1)
        assert snapshot.targets[AccountTargetKey("recording", IF)] == Decimal(0)

        # Runner默认使用自己的PositionManager作为策略仓位Provider。
        runner.position_manager.set_strategy_position("alpha", "rb", 4)
        assert runner.position("alpha", "rb") == Decimal(4)
    finally:
        runner.stop()
    print("阶段5/E4通过：Runner已接入目标存储、多策略净额和仓位查询")


def main() -> None:
    """按依赖顺序执行；任一阶段失败都会立即停止。"""

    test1_target_store()
    test2_portfolio_coordinator()
    test3_position_manager()
    test4_combined_flow()
    test5_runner_integration()
    print("Target/Portfolio/Position core tests OK")


if __name__ == "__main__":
    main()
