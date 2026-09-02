import os, pdb
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from pymongo import InsertOne, DeleteOne
from kdutil.mongodb import MongoDBManager
from datetime import datetime, date, timedelta
from toolix.macro.contract import OPTIONS_SPOT


@dataclass
class UnifiedDC:
    symbol: str  ## 标的
    spot_price: float  ## 现货价格
    strike_price: float  ## 行权价
    risk_free_rate: float  ## 无风险利率
    price: float  ## 期权的市场价格
    option_type: str  ## 期权类型
    american: str  ## 期权风格
    expiration_date: str  ## 到期日


class Signalor(object):

    def __init__(self, symbol, spot_symbol):
        self.symbol = symbol
        self._mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
        self._spot_symbol = spot_symbol

    def _get_third_friday(self, year, month):
        """计算某年某月的第三个星期五"""
        first_day_of_month = date(year, month, 1)
        first_friday_offset = (4 - first_day_of_month.weekday() + 7) % 7
        third_friday = first_day_of_month + timedelta(
            days=first_friday_offset + 14)
        return third_friday

    def fetch_spot_price(self, trade_time, pos):
        rt = self._mongo_client['neutron']['market_tick'].find({
            'symbol': self._spot_symbol,
            "datetime": {
                "$lte": trade_time
            }
        }).sort([("datetime", -1)]).limit(pos)
        data = pd.DataFrame(rt)
        if data.empty:
            return 0, None
        data = data.sort_values(by='datetime', ascending=False)
        return data.loc[0]['last_price'], data.loc[0]['datetime']

    def _parser_tick(self, spot_price, data):
        parts = data['symbol'].split('-')
        year = 2000 + int(parts[0][2:4])
        month = int(parts[0][4:])

        option_type = 'put' if parts[1] == 'P' else 'call'
        strike_price = float(parts[2])

        # 期权价格
        mid_price = (data['bid_price_1'] + data['ask_price_1']) / 2
        price = mid_price if mid_price > 0 else data['last_price']
        expiration_date = self._get_third_friday(year, month)
        is_american = True
        return UnifiedDC(symbol=data['symbol'],
                         spot_price=spot_price,
                         strike_price=strike_price,
                         risk_free_rate=0.04,
                         price=price,
                         option_type=option_type,
                         expiration_date=expiration_date.strftime('%Y-%m-%d'),
                         american=is_american)

    def _binomial_price(self, S, K, T, r, sigma, option_type, n):
        """美式期权定价（网页2优化模型）"""
        if T <= 0:  # 到期处理
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)

        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        disc = np.exp(-r * dt)

        # 向量化构建价格树（网页3性能优化）
        stock_tree = S * (u**np.arange(n, -1, -1)) * (d**np.arange(
            0, n + 1, 1))

        # 到期价值
        option_tree = np.maximum(
            stock_tree -
            K, 0) if option_type == 'call' else np.maximum(K - stock_tree, 0)

        # 逆向递归（网页4美式行权判断）
        for i in range(n - 1, -1, -1):
            stock_prices = S * (u**np.arange(i, -1, -1)) * (d**np.arange(
                0, i + 1, 1))
            hold_value = disc * (p * option_tree[:-1] +
                                 (1 - p) * option_tree[1:])
            exercise_value = stock_prices - K if option_type == 'call' else K - stock_prices
            option_tree = np.maximum(hold_value, exercise_value)

        return option_tree[0]

    def _calc_price(self, S, K, T, r, sigma, option_type, american, n_steps):
        """统一定价接口（网页1/2模型）"""
        if american:
            return self._binomial_price(S, K, T, r, sigma, option_type,
                                        n_steps)
        else:
            return self._bs_price(S, K, T, r, sigma, option_type)

    def _bs_price(self, S, K, T, r, sigma, option_type):
        """欧式期权定价（网页1公式）"""
        if T <= 0:  # 到期处理
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _calc_implied_vol(self, S, K, T, r, price, option_type, american,
                          n_steps):
        """整合波动率计算逻辑（网页1/2方法）"""

        def error_fn(sigma):
            return self._calc_price(S, K, T, r, sigma, option_type, american,
                                    n_steps) - price

        try:
            if american:
                # 二分法保证收敛（网页2建议）
                low, high = 0.01, 3.0
                for _ in range(100):
                    mid = (low + high) / 2
                    if error_fn(mid) > 0: high = mid
                    else: low = mid
                    if abs(high - low) < 1e-6: break
                return mid
            else:
                # 牛顿法快速求解（网页1方法）
                return newton(error_fn, x0=0.3, maxiter=100)
        except:
            return np.nan

    def _calculate_time_to_maturity(self, expiration_date, now):
        """计算剩余到期时间（年），精确到分钟级（网页5时间精度要求）"""
        #now = datetime.now()
        delta = expiration_date - datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
        total_seconds = max(delta.total_seconds(), 0)  # 防止负值
        return total_seconds / (365.25 * 24 * 3600)  # 精确年化

    def calculate_greeks(self,
                         S,
                         K,
                         T,
                         r,
                         iv,
                         option_type,
                         american,
                         n_steps=100):
        """根据已知iv，计算单个期权的Greeks。"""
        greeks = {}
        dS = S * 0.01
        dT = 1 / 365.25
        dIV = 0.01  # 1%的波动率变化
        dR = 0.01

        # 计算当前价格（虽然已有，但为了一致性可以重算）
        price = self._calc_price(S, K, T, r, iv, option_type, american,
                                 n_steps)

        # Delta & Gamma
        price_up = self._calc_price(S + dS, K, T, r, iv, option_type, american,
                                    n_steps)
        price_down = self._calc_price(S - dS, K, T, r, iv, option_type,
                                      american, n_steps)
        greeks['Delta'] = (price_up - price_down) / (2 * dS)
        greeks['Gamma'] = (price_up - 2 * price + price_down) / (dS**2)

        # Theta
        # 注意：Theta是时间流逝的损失，T减小。所以用 T-dT
        price_later = self._calc_price(S, K, T - dT, r, iv, option_type,
                                       american, n_steps)
        greeks['Theta'] = price_later - price

        # Vega (per 1% change in IV)
        price_iv_up = self._calc_price(S, K, T, r, iv + dIV, option_type,
                                       american, n_steps)
        greeks['Vega'] = (price_iv_up - price) / (dIV * 100)

        # Rho
        price_r_up = self._calc_price(S, K, T, r + dR, iv, option_type,
                                      american, n_steps)
        price_r_down = self._calc_price(S, K, T, r - dR, iv, option_type,
                                        american, n_steps)
        greeks['Rho'] = (price_r_up - price_r_down) / (2 * dR)

        return greeks

    def _create_greeks(self,
                       S,
                       K,
                       expiration_date,
                       r,
                       price,
                       option_type,
                       trade_time,
                       american=True,
                       n_steps=100):

        if isinstance(expiration_date, str):
            expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")

        pdb.set_trace()
        T = self._calculate_time_to_maturity(expiration_date, trade_time)
        ## 隐含波动率
        iv = self._calc_implied_vol(S, K, T, r, price, option_type, american,
                                    n_steps)

        greeks_data = self.calculate_greeks(S, K, T, r, iv, option_type, american)
        greeks_data = {
            'trade_time': trade_time,
            'symbol': self.symbol,
            'spot_symbol': self._spot_symbol,
            'S': S,
            'K': K,
            'expiration': expiration_date,
            'r': r,
            'price': price,
            'option_type': option_type,
            'american': 1 if american else 0,
            'n_steps': n_steps,
            'iv': iv
        }
        self.update_signalor(data=pd.DataFrame([greeks_data]),
                             table_name='tick_options_greeks')

    def update_signalor(self, data, table_name):
        insert_request = [
            InsertOne(data) for data in data.to_dict(orient='records')
        ]

        delete_request = [
            DeleteOne(data) for data in data[['trade_time', 'symbol']].to_dict(
                orient='records')
        ]
        _ = self._mongo_client['neutron'][table_name].bulk_write(
            delete_request + insert_request, bypass_document_validation=True)

    def run(self, trade_time):
        pass

    def on_tick(self, data):
        spot_price, spot_time = self.fetch_spot_price(
            trade_time=data['datetime'], pos=2)
        option_time = datetime.strptime(data['datetime'], "%Y-%m-%d %H:%M:%S")
        spot_time = datetime.strptime(spot_time, "%Y-%m-%d %H:%M:%S")
        time_diff = abs((option_time - spot_time).total_seconds())
        if spot_price == 0 or time_diff > 0.5:
            print(
                "error spot_price:{0}, spot_time:{1}, option_time:{2}".format(
                    spot_price, spot_time, option_time))
            #return
        unified_data = self._parser_tick(data=data, spot_price=spot_price)
        self._create_option(S=unified_data.spot_price,
                            K=unified_data.strike_price,
                            expiration_date=unified_data.expiration_date,
                            r=unified_data.risk_free_rate,
                            price=unified_data.price,
                            option_type=unified_data.option_type,
                            american=unified_data.american,
                            trade_time=data['datetime'])
