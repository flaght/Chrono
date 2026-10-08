"""Delta 研究信号：合约与参数、BS/Black76 定价、期限和虚值选约。"""
from dataclasses import dataclass
from datetime import date, datetime, time
from math import erf, exp, log, sqrt, pi, isfinite


@dataclass(frozen=True)
class Option:
    """研究合约条款；expiry 使用最后交易日判断可交易期限。"""
    symbol: str
    kind: str
    strike: float
    listed: date
    expiry: date
    tick: float
    multiplier: float
    month: str


@dataclass(frozen=True)
class Config:
    """选约与双腿买入参数；期限天数统一使用自然日。"""
    product: str = "MO"
    index_code: str = "000852"
    model: str = "bs"
    future_product: str = "IM"
    sides: tuple = ("C", "P")
    delta_min: float = .25
    delta_max: float = .30
    target_delta: float = .275
    fallback_min: float = .225
    fallback_max: float = .325
    allow_fallback: bool = True
    dte_min: int = 20
    dte_max: int = 45
    target_dte: float = 32.5
    min_remaining_days: int = 7
    expiry_month: str | None = None
    select_slots: tuple = ("13:58",)
    rate: float = .02
    dividend_yield: float = 0
    max_quote_age_seconds: int = 60
    min_open_interest: float = 1
    min_volume: float = 0
    quantity: int = 1
    close_remaining_days: int = 2
    flatten_slot: str = "14:50"
    max_market_age_seconds: int = 300

    def __post_init__(self):
        numbers=(self.delta_min,self.delta_max,self.target_delta,self.fallback_min,self.fallback_max,
                 self.target_dte,self.rate,self.dividend_yield,
                 self.min_open_interest,self.min_volume,self.dte_min,self.dte_max,
                 self.min_remaining_days,self.max_quote_age_seconds)
        if not all(isfinite(n) for n in numbers): raise ValueError("参数须为有限数值")
        if not 0 < self.fallback_min <= self.delta_min <= self.target_delta <= self.delta_max <= self.fallback_max < 1:
            raise ValueError("Delta目标、核心区间与回退区间无效")
        if not self.min_remaining_days > 0 or not self.min_remaining_days <= self.dte_min <= self.target_dte <= self.dte_max:
            raise ValueError("剩余自然日参数无效")
        if self.max_quote_age_seconds <= 0 or min(self.min_open_interest,self.min_volume)<0:
            raise ValueError("帧迟到限制或流动性参数无效")
        if self.model not in ("bs","black76") or not self.sides or len(set(self.sides))!=len(self.sides) or set(self.sides)-{"C","P"}:
            raise ValueError("模型或Call/Put类型无效")
        if self.expiry_month is not None:
            if len(self.expiry_month)!=6 or not self.expiry_month.isdigit(): raise ValueError("月份须为YYYYMM")
            datetime.strptime(self.expiry_month,"%Y%m")
        if not self.select_slots or len(set(self.select_slots))!=len(self.select_slots): raise ValueError("选约时点为空或重复")
        for slot in self.select_slots:
            if time.fromisoformat(slot).strftime("%H:%M")!=slot: raise ValueError("选约时点须为HH:MM")
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool) or self.quantity <= 0:
            raise ValueError("每腿数量须为正整数")
        if not isinstance(self.close_remaining_days, int) or not 0 <= self.close_remaining_days < self.min_remaining_days:
            raise ValueError("退出剩余自然日须非负且小于最低开仓剩余自然日")
        if not isfinite(self.max_market_age_seconds) or self.max_market_age_seconds <= 0:
            raise ValueError("行情年龄限制须为有限正数")
        if time.fromisoformat(self.flatten_slot).strftime("%H:%M") != self.flatten_slot:
            raise ValueError("清仓时点须为HH:MM")


def years_to_expiry(now, expiry):
    """到最后交易日 15:00 的自然年化剩余时间。"""
    return max(0,(datetime.combine(expiry,time(15),tzinfo=now.tzinfo)-now).total_seconds()/(365*86400))


def cdf(x): return (1+erf(x/sqrt(2)))/2


@dataclass(frozen=True)
class Greeks:
    iv: float
    delta: float
    gamma: float
    vega: float


def price(x,k,t,r,q,vol,kind,model):
    """BS 使用指数现价，Black76 使用同月期货，返回理论权利金。"""
    if min(x,k,t,vol)<=0: raise ValueError("定价输入非正")
    forward=x if model=="black76" else x*exp((r-q)*t)
    root=vol*sqrt(t)
    d1=log(forward/k)/root+root/2
    d2=d1-root
    if kind=="C": return exp(-r*t)*(forward*cdf(d1)-k*cdf(d2))
    return exp(-r*t)*(k*cdf(-d2)-forward*cdf(-d1))


def implied_greeks(x,k,t,r,q,premium,kind,model):
    """检查无套利边界，二分反解 IV，再计算模型对应的风险指标。"""
    if not all(isfinite(n) for n in (x,k,t,r,q,premium)) or min(x,k,t,premium)<=0: return None
    forward=x if model=="black76" else x*exp((r-q)*t)
    disc=exp(-r*t)
    lower=disc*max((forward-k) if kind=="C" else (k-forward),0)
    upper=disc*(forward if kind=="C" else k)
    if not lower<premium<upper: return None
    lo,hi=1e-5,5.
    if not price(x,k,t,r,q,lo,kind,model)<=premium<=price(x,k,t,r,q,hi,kind,model): return None
    for _ in range(80):
        mid=(lo+hi)/2
        if price(x,k,t,r,q,mid,kind,model)>premium: hi=mid
        else: lo=mid
    iv=(lo+hi)/2
    d1=log(forward/k)/(iv*sqrt(t))+iv*sqrt(t)/2
    pdf=exp(-d1*d1/2)/sqrt(2*pi)
    factor=disc if model=="black76" else exp(-q*t)
    delta=factor*(cdf(d1) if kind=="C" else cdf(d1)-1)
    gamma=factor*pdf/(x*iv*sqrt(t))
    vega=disc*forward*pdf*sqrt(t)
    return Greeks(iv,delta,gamma,vega)


def select(options, quotes, index, forward_by_month, now, config):
    """先确定期限月份，再分别选 Call/Put，返回结果、审计与选定月份。"""
    rejected=[]
    def reject(symbol,reason): rejected.append({"symbol":symbol,"reason":reason})
    if index is None or not isfinite(index) or index<=0:
        return [],[{"symbol":config.index_code,"reason":"MISSING_INDEX"}],None
    months={}
    for option in options.values():
        dte=(option.expiry-now.date()).days
        if option.kind not in config.sides or not option.listed<=now.date()<=option.expiry: continue
        if dte<config.min_remaining_days: continue
        if config.expiry_month:
            if option.month!=config.expiry_month: continue
        elif not config.dte_min<=dte<=config.dte_max: continue
        months[option.month]=(option.expiry,dte)
    if not months: return [],[{"symbol":"","reason":"NO_ELIGIBLE_EXPIRY"}],None
    # 只按期限确定月份；不使用其他月份的质量/Delta结果反向选择期限。
    month=min(months,key=lambda m:(abs(months[m][1]-config.target_dte),months[m][0],m))
    x=index if config.model=="bs" else forward_by_month.get(month)
    if x is None or not isfinite(x) or x<=0:
        return [],[{"symbol":month,"reason":"MISSING_SAME_MONTH_FORWARD"}],month
    qualified=[]
    for option in options.values():
        if option.month!=month or option.kind not in config.sides or not option.listed<=now.date()<=option.expiry: continue
        if (option.kind=="C" and option.strike<=index) or (option.kind=="P" and option.strike>=index):
            reject(option.symbol,"NOT_OTM"); continue
        quote=quotes.get(option.symbol)
        if quote is None: reject(option.symbol,"MISSING_BAR"); continue
        if "bar_ns" not in quote or not isfinite(quote["bar_ns"]):
            reject(option.symbol,"MISSING_BAR_TIME"); continue
        if quote["bar_ns"]!=int(now.timestamp()*1e9): reject(option.symbol,"MISMATCHED_BAR_TIME"); continue
        fields=("close","open_interest","volume")
        if any(key not in quote or not isfinite(quote[key]) for key in fields): reject(option.symbol,"MISSING_BAR_FIELDS"); continue
        if any(quote[key]<0 for key in ("open_interest","volume")):
            reject(option.symbol,"INVALID_BAR_FIELDS"); continue
        close=quote["close"]
        if close<=0: reject(option.symbol,"INVALID_CLOSE"); continue
        if quote["open_interest"]<config.min_open_interest or quote["volume"]<config.min_volume:
            reject(option.symbol,"LOW_LIQUIDITY"); continue
        g=implied_greeks(x,option.strike,years_to_expiry(now,option.expiry),config.rate,config.dividend_yield,close,option.kind,config.model)
        if g is None: reject(option.symbol,"INVALID_IV_OR_ARBITRAGE_BOUND"); continue
        qualified.append(dict(symbol=option.symbol,kind=option.kind,month=month,expiry=str(option.expiry),
            strike=option.strike,dte=months[month][1],index=index,model_input=x,model=config.model,
            rate=config.rate,dividend_yield=config.dividend_yield,close=close,bar_ns=quote["bar_ns"],price_source="close",
            iv=g.iv,delta=g.delta,gamma=g.gamma,vega=g.vega,distance=abs(abs(g.delta)-config.target_delta),
            open_interest=quote["open_interest"],volume=quote["volume"],zero_volume_bar=quote["volume"]==0))
    chosen=[]
    for kind in config.sides:
        candidates=[row for row in qualified if row["kind"]==kind and config.delta_min<=abs(row["delta"])<=config.delta_max]
        fallback=False
        if not candidates and config.allow_fallback:
            candidates=[row for row in qualified if row["kind"]==kind and config.fallback_min<=abs(row["delta"])<=config.fallback_max]
            fallback=True
        if candidates:
            best=min(candidates,key=lambda row:(row["distance"],-row["open_interest"],-row["volume"],row["symbol"]))
            chosen.append({**best,"fallback":fallback})
        else: reject(kind,"NO_DELTA_CANDIDATE")
    for row in qualified:
        rejected.append({**row,"reason":"CANDIDATE","selected":any(r["symbol"]==row["symbol"] for r in chosen)})
    return chosen,rejected,month
