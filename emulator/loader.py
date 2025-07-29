import pdb, os
import pandas as pd
from object import TickData, BarData, Interval
from constant import MAPPING


class RQConverter(object):

    @classmethod
    def dataframe(cls, data):
        return [
            TickData(code=row.code,
                     symbol=row.symbol,
                     exchange=MAPPING[row.code],
                     create_time=row.trade_time,
                     volume=row.volume,
                     turnover=row.total_turnover,
                     open_interest=row.open_interest,
                     last_price=row.last,
                     limit_up=row.limit_up,
                     limit_down=row.limit_down,
                     open_price=row.open,
                     high_price=row.high,
                     low_price=row.low,
                     bid_price_1=row.b1,
                     bid_price_2=row.b2,
                     bid_price_3=row.b3,
                     bid_price_4=row.b4,
                     bid_price_5=row.b5,
                     ask_price_1=row.a1,
                     ask_price_2=row.a2,
                     ask_price_3=row.a3,
                     ask_price_4=row.a4,
                     ask_price_5=row.a5,
                     bid_volume_1=row.b1_v,
                     bid_volume_2=row.b2_v,
                     bid_volume_3=row.b3_v,
                     bid_volume_4=row.b4_v,
                     bid_volume_5=row.b5_v,
                     ask_volume_1=row.a1_v,
                     ask_volume_2=row.a2_v,
                     ask_volume_3=row.a3_v,
                     ask_volume_4=row.a4_v,
                     ask_volume_5=row.a5_v) for row in data.itertuples()
        ]


class UQer(object):

    @classmethod
    def dataframe(cls, data):
        return BarData(code=data.loc[0]['code'],
                       exchange=MAPPING[data.loc[0]['code']],
                       symbol=data.loc[0]['symbol'],
                       create_time=data.loc[0]['trade_date'],
                       interval=Interval.DAILY,
                       volume=data.loc[0]['volume'],
                       turnover=data.loc[0]['turnover'],
                       open_interest=data.loc[0]['open_interest'],
                       open_price=data.loc[0]['open_price'],
                       high_price=data.loc[0]['high_price'],
                       low_price=data.loc[0]['low_price'],
                       close_price=data.loc[0]['close_price'],
                       settle_price=data.loc[0]['settle_price'])


class Loader(object):

    @classmethod
    def create_loader(cls, uri, types='file'):
        if types == 'file':
            return FileLoader(uri=uri)


class FileLoader(Loader):
    __name__ = "file"

    def __init__(self, uri):
        self.base_dir = uri

    def fetch_ticket(self, **kwargs):
        data = pd.read_feather(
            os.path.join(self.base_dir, 'tick', kwargs['code'],
                         "{0}.feather".format(kwargs['trade_date'])))
        market_tick_list = RQConverter.dataframe(data=data)
        return market_tick_list

    def fetch_daily(self, **kwargs):
        data = pd.read_feather(
            os.path.join(self.base_dir, 'daily', kwargs['code'],
                         "{0}.feather".format(kwargs['trade_date'])))

        market_daily = UQer.dataframe(data=data)
        pdb.set_trace()
        return market_daily
