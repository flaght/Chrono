import pdb
from ultron.tradingday import *
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


def create_features(begin_date, end_date, code='000852'):
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
    total_data = smart_money_data.reset_index().merge(
        sentiment_shock_data.reset_index(),
        on=['trade_date']).merge(mean_reversion_data.reset_index()).merge(
            trend_momentum_data.reset_index(), on=['trade_date'])
    return total_data[total_data['trade_date']<=end_date]
