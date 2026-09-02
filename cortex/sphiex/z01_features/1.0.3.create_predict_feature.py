import pdb, itertools, os, toml, asyncio, math
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar
from kdutils.macro import base_path
from kdutils.ttimes import get_dates
from features.predict import MeanReversion
from features.predict import SentimentShock
from features.predict import SmartMoney
from features.predict import TrendMomentum


def create_mean_reversion(begin_date, end_date, code):
    factor1 = MeanReversion(code=code)
    return factor1.start(begin_date=begin_date, end_date=end_date)


def create_sentiment_shock(begin_date, end_date, code):
    factor1 = SentimentShock(code=code)
    return factor1.start(begin_date=begin_date, end_date=end_date)


def create_smart_money(begin_date, end_date, code):
    factor1 = SmartMoney(code=code)
    return factor1.start(begin_date=begin_date, end_date=end_date)


def create_trend_momentum(begin_date, end_date, code):
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-60b').strftime('%Y-%m-%d')
    factor1 = TrendMomentum(code=code)
    return factor1.start(begin_date=start_date, end_date=end_date)


def create_data(method, code='000852'):
    pdb.set_trace()
    begin_date, end_date = get_dates(method=method)
    begin_date1 = advanceDateByCalendar('china.sse', begin_date,
                                        '-30b').strftime('%Y-%m-%d')
    end_date1 = advanceDateByCalendar('china.sse', end_date,
                                      '{0}b'.format(5)).strftime('%Y-%m-%d')

    trend_momentum_data = create_trend_momentum(begin_date=begin_date1,
                                                end_date=end_date1,
                                                code=code)
    smart_money_data = create_smart_money(begin_date=begin_date1,
                                          end_date=end_date1,
                                          code=code)
    sentiment_shock_data = create_sentiment_shock(begin_date=begin_date1,
                                                  end_date=end_date1,
                                                  code=code)
    mean_reversion_data = create_mean_reversion(begin_date=begin_date1,
                                                end_date=end_date1,
                                                code=code)
    pdb.set_trace()
    total_data = smart_money_data.reset_index().merge(
        sentiment_shock_data.reset_index(),
        on=['trade_date']).merge(mean_reversion_data.reset_index()).merge(
            trend_momentum_data.reset_index(), on=['trade_date'])
    # total_data = total_data[(total_data['trade_date'] >= begin_date)
    #                         & (total_data['trade_date'] <= end_date)]
    pdb.set_trace()
    base_dir = os.path.join("records", "basic", method)
    os.makedirs(base_dir, exist_ok=True)
    total_data.reset_index(drop=True).to_feather(
        os.path.join(base_dir, "predict_data.feather"))


if __name__ == '__main__':
    method = 'train0'
    create_data(method=method)
