"""显式场景组装函数，不自动注册策略。"""
from .fixed_contracts import plan_fixed_contracts, prepare_fixed_contracts
from .option_chain import plan_option_chain, prepare_option_chain
from .role_futures import prepare_role_futures, prepare_role_research, plan_role_futures

__all__ = ["plan_fixed_contracts", "prepare_fixed_contracts", "plan_option_chain",
           "prepare_option_chain", "plan_role_futures", "prepare_role_futures", "prepare_role_research"]
