import itertools, pdb
from ultron.sentry.api import *
from lumina.evolution.fusion.actuator import Actuator
from lib.attr001.integrity.impulse import run_factors1, diagnostics1


def run_factors2(factors_infos, market_data):
    total_data = run_factors1(factors_infos, market_data)
    actuator = Actuator(k_split=1)
    original_factors, normal_factors = actuator.calculate(
        factors_infos=factors_infos,
        total_data=total_data.reset_index(),
        method='roll_zscore',
        win=15)
    return original_factors, normal_factors


def diagnostics(factors_infos, research_data, trader_data, top_n=10):
    research_original_factors, research_normal_factors = run_factors2(
        factors_infos=factors_infos, market_data=research_data)
    trader_original_factors, trader_normal_factors = run_factors2(
        factors_infos=factors_infos, market_data=trader_data)

    original_metrics = diagnostics1(research_factors=research_original_factors,
                                    trader_factors=trader_original_factors,
                                    top_n=top_n)

    normal_metrics = diagnostics1(research_factors=research_normal_factors,
                                  trader_factors=trader_normal_factors,
                                  top_n=top_n)

    return original_metrics, normal_metrics
