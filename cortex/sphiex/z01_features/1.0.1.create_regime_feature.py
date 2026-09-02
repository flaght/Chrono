import pdb, itertools, os, toml, asyncio, math
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from dotenv import load_dotenv

load_dotenv()
#from jdw import DBAPI
from ultron.tradingday import *
from kdutils.macro import base_path
from kdutils.ttimes import get_dates
from features.regime import MacroFeature
from features.regime import PriceFeature
from features.regime import VolatilityFeature
from features.regime import BreadthFeature
from features.regime import FundamentalFeature
from features.regime import OverviewFeature


def create_macroe(begin_date, end_date):
    factor1 = MacroFeature()
    return factor1.start(begin_date=begin_date, end_date=end_date)


def create_price(begin_date, end_date, code):
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-80b').strftime('%Y-%m-%d')
    factor1 = PriceFeature(code=code)
    return factor1.start(begin_date=start_date, end_date=end_date)


def create_volatility(begin_date, end_date, code):
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-252b').strftime('%Y-%m-%d')
    factor1 = VolatilityFeature(code=code)
    return factor1.start(begin_date=start_date, end_date=end_date)


def create_breadth(begin_date, end_date, code):
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-252b').strftime('%Y-%m-%d')
    factor1 = BreadthFeature(code=code)
    return factor1.start(begin_date=start_date, end_date=end_date)


def create_fundamental(begin_date, end_date, code):
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-70b').strftime('%Y-%m-%d')
    factor1 = FundamentalFeature(code=code)
    return factor1.start(begin_date=start_date, end_date=end_date)


def create_overview(begin_date, end_date, code):
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-252b').strftime('%Y-%m-%d')
    factor1 = OverviewFeature(code=code)
    return factor1.start(begin_date=start_date, end_date=end_date)


def create_data(method, code='000852'):
    begin_date, end_date = get_dates(method=method)
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-30b').strftime('%Y-%m-%d')
    overview_data = create_overview(begin_date=start_date,
                                    end_date=end_date,
                                    code=code)
    price_data = create_price(begin_date=start_date,
                              end_date=end_date,
                              code=code)

    fundamental_data = create_fundamental(begin_date=start_date,
                                          end_date=end_date,
                                          code=code)

    breadth_data = create_breadth(begin_date=start_date,
                                  end_date=end_date,
                                  code=code)
    volatility_data = create_volatility(begin_date=start_date,
                                        end_date=end_date,
                                        code=code)
    macroe_data = create_macroe(begin_date=start_date, end_date=end_date)

    total_data = overview_data.reset_index().merge(
        price_data.stack().reset_index(), on=['trade_date']).merge(
            breadth_data.reset_index(),
            on=['trade_date']).merge(macroe_data, on=['trade_date']).merge(
                volatility_data.reset_index(),
                on=['trade_date','code']).merge(fundamental_data.reset_index(),
                                         on=['trade_date'])
    base_dir = os.path.join("records", "basic", method)
    os.makedirs(base_dir, exist_ok=True)
    pdb.set_trace()
    total_data.to_feather(os.path.join(base_dir, "regime_data.feather"))


if __name__ == '__main__':
    method = 'train0'
    create_data(method=method)
