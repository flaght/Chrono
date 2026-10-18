from collections import namedtuple


class OrderTuple(
        namedtuple(
            'OrderTuple',
            ('buy_time', 'buy_price', 'buy_cnt', 'sell_time', 'sell_price',
             'sell_type', 'expect_direction', 'buy_symbol'))):
    __slots__ = ()
