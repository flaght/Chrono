"""期权条款、配置、日期资格与认购隐含波动率、风险指标和卖方配仓。"""
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from math import erf, exp, floor, isfinite, log, pi, sqrt
from typing import Mapping


@dataclass(frozen=True)
class Contract:
    """认购期权静态条款与对应同月对冲期货，不负责读取文件。"""
    symbol: str
    future: str
    strike: float
    multiplier: Decimal
    list_day: date
    last_day: date
    expiry_day: date
    tick: Decimal


@dataclass(frozen=True)
class OptionVegaConfig:
    """固定规模的波动率风险预算、每日时点、退出规则和手数限制。"""
    nav: Decimal = Decimal("40000000")
    vega_budget: float = 1.25
    delta_targets: tuple[float, ...] = (0.2, 0.25, 0.3, 0.35)
    option_slot: str = "13:58"
    hedge_slots: tuple[str, ...] = ("10:00", "11:00", "13:30", "14:00", "14:55")
    flatten_slot: str = "14:50"
    min_remaining_days: int = 5
    close_remaining_days: int = 2
    rate: float = 0.02
    min_time_value: float = 5.0
    max_market_age_seconds: int = 300
    max_iv_age_seconds: int = 300
    max_option_lots: int = 100
    max_future_lots: int = 100
    max_total_option_lots: int = 200

    def __post_init__(self):
        nav = Decimal(str(self.nav))
        if not nav.is_finite() or nav <= 0:
            raise ValueError("nav必须为有限正数")
        object.__setattr__(self, "nav", nav)
        if not isfinite(self.vega_budget) or self.vega_budget <= 0:
            raise ValueError("vega_budget必须为有限正数")
        if not self.delta_targets or any(not isfinite(v) or not 0 < v < 1 for v in self.delta_targets):
            raise ValueError("delta_targets须在(0,1)范围内")
        if not isfinite(self.rate) or not isfinite(self.min_time_value) or self.min_time_value < 0:
            raise ValueError("利率或最小时间价值无效")
        for name in ("max_market_age_seconds", "max_iv_age_seconds", "max_option_lots", "max_future_lots", "max_total_option_lots"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name}必须为正整数")
        if not 0 <= self.close_remaining_days < self.min_remaining_days:
            raise ValueError("需满足0 <= close_remaining_days < min_remaining_days")
        for slot in (self.option_slot, self.flatten_slot, *self.hedge_slots):
            if time.fromisoformat(slot).strftime("%H:%M") != slot:
                raise ValueError("时点必须为HH:MM")
        if self.option_slot >= self.flatten_slot:
            raise ValueError("开仓时点必须早于最终清仓时点")

    @property
    def vega_cash_per_unit(self):
        # 数值函数的敏感度按波动率变化1计算，预算对应规模基准乘预算参数再除以100。
        # 波动率变化一个百分点时的现金预算还需除以100，即规模基准乘参数除以10000。
        return float(self.nav) * self.vega_budget / 100


def tau(now: datetime, last_day: date) -> float:
    # 隐含波动率和风险指标统一使用自然年化剩余时间，不混用交易年与自然年。
    expiry = datetime.combine(last_day, time(15), tzinfo=now.tzinfo)
    return max((expiry - now).total_seconds() / (365 * 86400), 0.0)


def remaining_trading_days(trading_days, day, last_day, threshold):
    """统计不含当日、含最后交易日的日期；远期只需足以判断策略阈值。

    若到期超过已知日期末端，计数是下界。下界达到threshold时可以确定
    尚未进入对应开仓/清仓限制；否则不把缺少未来日期当成临近到期。
    """
    if not trading_days:
        raise ValueError("没有可用交易日")
    if last_day <= trading_days[-1] and last_day not in trading_days:
        raise ValueError(f"合约最后交易日{last_day}不在已知交易日中")
    count = bisect_right(trading_days, last_day) - bisect_right(trading_days, day)
    if last_day > trading_days[-1] and count < threshold:
        raise ValueError(
            f"{day}: 合约最后交易日{last_day}超出已知日期末端{trading_days[-1]}，"
            f"之后仅有{count}个已知交易日，无法判断{threshold}日阈值；"
            "请让行情日期覆盖回测结束后的足够交易日，或提供--calendar"
        )
    return count


def cdf(x):
    """标准正态分布的累计概率。"""
    return (1 + erf(x / sqrt(2))) / 2


def call_price(forward, strike, years, rate, volatility):
    """以同月期货价格为标的，计算认购期权理论价格。"""
    if not all(isfinite(v) for v in (forward, strike, years, rate, volatility)) or min(forward, strike, years, volatility) <= 0:
        raise ValueError("Black76输入无效")
    root = volatility * sqrt(years)
    d1 = log(forward / strike) / root + root / 2
    return exp(-rate * years) * (forward * cdf(d1) - strike * cdf(d1 - root))


@dataclass(frozen=True)
class Greeks:
    """保存隐含波动率及价格一阶、二阶和波动率敏感度。"""
    iv: float
    delta: float
    gamma: float
    vega: float


def implied_greeks(forward, strike, years, rate, premium):
    """校验价格边界并二分反解隐含波动率，失败返回空值。"""
    if not all(isfinite(v) for v in (forward, strike, years, rate, premium)) or min(forward, strike, years, premium) <= 0:
        return None
    discount = exp(-rate * years)
    if not discount * max(forward - strike, 0) < premium < discount * forward:
        return None
    lo, hi = 0.00001, 5.0
    if not call_price(forward, strike, years, rate, lo) <= premium <= call_price(forward, strike, years, rate, hi):
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if call_price(forward, strike, years, rate, mid) > premium:
            hi = mid
        else:
            lo = mid
    iv = (lo + hi) / 2
    if abs(call_price(forward, strike, years, rate, iv) - premium) > max(1e-8, premium * 1e-8):
        return None
    d1 = (log(forward / strike) + iv * iv * years / 2) / (iv * sqrt(years))
    pdf = exp(-d1 * d1 / 2) / sqrt(2 * pi)
    return Greeks(iv, discount * cdf(d1), discount * pdf / (forward * iv * sqrt(years)),
                  discount * forward * pdf * sqrt(years))


def cached_greeks(forward, strike, years, rate, iv):
    """使用仍在有效期内的缓存波动率，按当前价格和期限重算风险指标。"""
    if min(forward, strike, years, iv) <= 0:
        return None
    return implied_greeks(forward, strike, years, rate, call_price(forward, strike, years, rate, iv))


def allocate_short(candidates, delta_targets, cash_vega_budget, max_lots, max_total):
    """候选元组(symbol, Greeks, multiplier)；同一合约可对应多个Delta档。

    向下取整；合并重复合约。单次组合现金Vega不超过预算，不每日累加。
    """
    if not candidates:
        return {}
    choices = [min(candidates, key=lambda x: (abs(x[1].delta - target), x[0])) for target in delta_targets]
    weights = [1 / (item[1].gamma * float(item[2])) for item in choices]
    denominator = sum(w * item[1].vega * float(item[2]) for w, item in zip(weights, choices))
    raw = {}
    for weight, item in zip(weights, choices):
        raw[item[0]] = raw.get(item[0], 0) + cash_vega_budget * weight / denominator
    quantities = {key: min(floor(value + 1e-10), max_lots) for key, value in raw.items()}
    total = sum(quantities.values())
    if total > max_total:
        quantities = {key: floor(value * max_total / total) for key, value in quantities.items()}
    return {key: -value for key, value in quantities.items() if value > 0}


def filter_hedgeable_options(candidates, contracts, future_rows, day):
    """剔除同月期货缺条款、未上市或已到期的认购候选。"""
    eligible, skipped = [], set()
    for key in candidates:
        future = contracts[key].future
        if future not in future_rows:
            skipped.add(future)
            continue
        row = future_rows[future]
        if not row["listDate"] <= day <= row["lastTradeDate"]:
            skipped.add(future)
            continue
        eligible.append(key)
    return tuple(eligible), tuple(sorted(skipped))
