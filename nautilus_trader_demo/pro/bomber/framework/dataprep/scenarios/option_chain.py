"""期权研究与执行共用基础数据，选约规则由策略负责。"""
from ..bars import read_bar_frame
from ..catalog import scan_bar_files, select_bar_files
from ..contracts import BarReadSpec, HistoricalInputBundle, InputContext, InputPlan, SourceSpec

OPTION_COLUMNS = ("close", "open_interest", "volume")


def plan_option_chain(required, optional=(), *, requested=(None, None), bindings=()):
    return InputPlan(tuple(sorted(set(required))), tuple(sorted(set(optional))), requested, tuple(bindings))


def read_execution_frame(path, expected_symbol, day, bar_timestamp, *, asset_kind="option"):
    from ..contracts import BarFileKey
    spec = BarReadSpec(timestamp_label=bar_timestamp, trading_day_policy="day_session")
    return read_bar_frame(path, spec, BarFileKey(asset_kind, expected_symbol, day)).frame


def read_research_frame(path, key, day, kind, bar_timestamp):
    from ..contracts import BarFileKey
    if kind not in {"option", "future", "index"}:
        raise ValueError("Unknown research asset kind")
    spec = BarReadSpec(required_fields=OPTION_COLUMNS if kind == "option" else ("close",),
        timestamp_label=bar_timestamp, trading_day_policy="day_session",
        value_policy="research_audited" if kind == "option" else "close_strict",
        # 保留现有研究拒绝阈值；研究数据中的小数成交量
        # 不额外套用执行场景的整数成交量约束。
        require_integer_volume=False)
    return read_bar_frame(path, spec, BarFileKey(kind, key, day)).frame


def prepare_option_chain(paths, plan, *, specs=None, context=None):
    index = {}
    for kind, root in paths.items():
        index.update(scan_bar_files(root, kind))
    files, coverage = select_bar_files(index, plan.required, plan.optional, plan.requested)
    specs = specs or {}
    sources = []
    for key, path in sorted(files.items()):
        spec = specs.get(key.asset_kind, BarReadSpec(trading_day_policy="day_session"))
        purpose = "research" if spec.value_policy == "research_audited" else "execution"
        sources.append(SourceSpec(read_bar_frame(path, spec, key), purpose))
    return HistoricalInputBundle(context or InputContext(), tuple(sources), plan.bindings, coverage)
