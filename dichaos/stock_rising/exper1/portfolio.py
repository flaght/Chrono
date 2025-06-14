import pdb
from datetime import date


class Portflio(object):

    def __init__(self, lookback_window_size: int = 0) -> None:
        self.price_series = {}
        self.returns_series = {}
        self.actions_series = {}
        self.share_series = {}
        self.day_count = 0
        self.holding_shares = 0
        self.lookback_window_size = lookback_window_size

    def transform(self, returns: dict):
        actions = [{
            'code': r['code'],
            'return': r['return'],
            'action': 1 if r['return'] > 0 else -1
        } for r in returns]
        return actions

    def update_market_info(self, cur_date: date, market_price: dict,
                           returns: dict):
        self.price_series[cur_date] = market_price
        self.returns_series[cur_date] = returns
        self.cur_date = cur_date
        self.day_count += 1

    def record_action(self, cur_date: date, actions: list[dict]):
        self.actions_series[cur_date] = [{
            'code': action['code'],
            'action': action['action']
        } for action in actions]
        self.share_series[cur_date] = [{
            'code': action['code'],
            'action': action['action']
        } for action in actions]

    def feedback(self, cur_date: date):
        ## 方向和收益率
        #if self.day_count <= self.lookback_window_size:
        #    return {}
        feedback_sets = {}
        share_series = self.share_series[cur_date]
        share_series = dict(
            zip([ss['code'] for ss in share_series],
                [ss['action'] for ss in share_series]))
        return_series = self.returns_series[cur_date]

        for k, v in return_series.items():
            returns = v
            feedback = 0
            share = share_series[k] if k in share_series else 0
            temp = returns * share
            if temp > 0:
                feedback = 1
            elif temp < 0:
                feedback = -1
            feedback_sets[k] = {'feedback': feedback, "date": cur_date, "code":k}

        return feedback_sets
