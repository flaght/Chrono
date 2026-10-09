"""准备声明的真实合约，不选择交易角色。"""
from ..bars import read_bar_frame
from ..catalog import scan_bar_files, select_bar_files
from ..contracts import BarReadSpec, HistoricalInputBundle, InputContext, InputPlan, SourceSpec


def plan_fixed_contracts(required, optional=(), *, requested=(None, None), bindings=()):
    return InputPlan(tuple(sorted(set(required))), tuple(sorted(set(optional))), requested, tuple(bindings))


def prepare_fixed_contracts(root, plan, *, spec=None, context=None):
    index = scan_bar_files(root, "future")
    files, coverage = select_bar_files(index, plan.required, plan.optional, plan.requested)
    sources = tuple(SourceSpec(read_bar_frame(path, spec or BarReadSpec(), key))
                    for key, path in sorted(files.items()))
    coverage.assumptions += ("required_files_are_the_callers_declared_set_not_a_full_exchange_calendar",)
    coverage.files = {str(source.result.path): {"rows": len(source.result.frame),
        "first_ns": source.result.first_ns, "last_ns": source.result.last_ns} for source in sources}
    return HistoricalInputBundle(context or InputContext(), sources, plan.bindings, coverage)
