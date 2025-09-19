
class BarGenerator:
    '''
    on_bar:当一根新的1分钟K线合成完毕时,将合成好的 BarData 对象作为参数传给它
    window:在已经合成1分钟基础上，在合成window 周期
    on_window_bar: 回调函数。如果设置了 window > 1，那么当一根X周期的K线合成完毕时，BarGenerator会调用这个 on_window_bar 函数
    interval:指定了 window 的单位，是分钟 (Interval.MINUTE)、小时 (Interval.HOUR) 还是天 (Interval.DAILY)。
    daily_end:仅在合成日线时使用，用于指定每日的收盘时间（例如 time(15, 0)），以确定一根日K线何时结束
    '''

    def __init__(self,
                 on_bar: Callable,
                 window: int = 0,
                 on_window_bar: Callable = None,
                 interval: Interval = Interval.MINUTE,
                 daily_end: time = None) -> None:
        self.bar: BarData = None
        self.on_bar: Callable = on_bar

        self.interval: Interval = interval
        self.interval_count: int = 0

        self.hour_bar: BarData = None
        self.daily_bar: BarData = None

        self.window: int = window
        self.window_bar: BarData = None
        self.on_window_bar: Callable = on_window_bar

        self.last_tick: TickData = None

        self.daily_end: time = daily_end
        if self.interval == Interval.DAILY and not self.daily_end:
            raise RuntimeError("合成日K线必须传入每日收盘时间")

    def update_tick(self, tick: TickData) -> None:
        """
        Update new tick data into generator.
        """
        new_minute: bool = False
        if not tick.last_price:
            return
        if not self.bar:
            new_minute = True
        elif ((self.bar.datetime.minute != tick.datetime.minute)
              or (self.bar.datetime.hour != tick.datetime.hour)):
            self.bar.datetime = self.bar.datetime.replace(second=0,
                                                          microsecond=0)
            self.on_bar(self.bar)
            new_minute = True

        if new_minute:
            self.bar = BarData(symbol=tick.symbol,
                               exchange=tick.exchange,
                               interval=Interval.MINUTE,
                               datetime=tick.datetime,
                               gateway_name=tick.gateway_name,
                               open_price=tick.last_price,
                               high_price=tick.last_price,
                               low_price=tick.last_price,
                               close_price=tick.last_price,
                               open_interest=tick.open_interest)
        elif self.bar:
            self.bar.high_price = max(self.bar.high_price, tick.last_price)
            if self.last_tick and tick.high_price > self.last_tick.high_price:
                self.bar.high_price = max(self.bar.high_price, tick.high_price)

            self.bar.low_price = min(self.bar.low_price, tick.last_price)
            if self.last_tick and tick.low_price < self.last_tick.low_price:
                self.bar.low_price = min(self.bar.low_price, tick.low_price)

            self.bar.close_price = tick.last_price
            self.bar.open_interest = tick.open_interest
            self.bar.datetime = tick.datetime

        if self.last_tick and self.bar:
            volume_change: float = tick.volume - self.last_tick.volume
            self.bar.volume += max(volume_change, 0)

            turnover_change: float = tick.turnover - self.last_tick.turnover
            self.bar.turnover += max(turnover_change, 0)

        self.last_tick = tick

    def update_bar(self, bar: BarData) -> None:
        """
        Update 1 minute bar into generator
        """
        if self.interval == Interval.MINUTE:
            self.update_bar_minute_window(bar)
        elif self.interval == Interval.HOUR:
            self.update_bar_hour_window(bar)
        else:
            self.update_bar_daily_window(bar)

    def update_bar_minute_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create window bar object
        if not self.window_bar:
            dt: datetime = bar.datetime.replace(second=0, microsecond=0)
            self.window_bar = BarData(symbol=bar.symbol,
                                      exchange=bar.exchange,
                                      datetime=dt,
                                      gateway_name=bar.gateway_name,
                                      open_price=bar.open_price,
                                      high_price=bar.high_price,
                                      low_price=bar.low_price)
        # Otherwise, update high/low price into window bar
        else:
            self.window_bar.high_price = max(self.window_bar.high_price,
                                             bar.high_price)
            self.window_bar.low_price = min(self.window_bar.low_price,
                                            bar.low_price)

        # Update close price/volume/turnover into window bar
        self.window_bar.close_price = bar.close_price
        self.window_bar.volume += bar.volume
        self.window_bar.turnover += bar.turnover
        self.window_bar.open_interest = bar.open_interest

        # Check if window bar completed
        if not (bar.datetime.minute + 1) % self.window:
            if self.on_window_bar:
                self.on_window_bar(self.window_bar)

            self.window_bar = None

    def update_bar_hour_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create window bar object
        if not self.hour_bar:
            dt: datetime = bar.datetime.replace(minute=0,
                                                second=0,
                                                microsecond=0)
            self.hour_bar = BarData(symbol=bar.symbol,
                                    exchange=bar.exchange,
                                    datetime=dt,
                                    gateway_name=bar.gateway_name,
                                    open_price=bar.open_price,
                                    high_price=bar.high_price,
                                    low_price=bar.low_price,
                                    close_price=bar.close_price,
                                    volume=bar.volume,
                                    turnover=bar.turnover,
                                    open_interest=bar.open_interest)
            return

        finished_bar: BarData | None = None

        # If minute is 59, update minute bar into window bar and push
        if bar.datetime.minute == 59:
            self.hour_bar.high_price = max(self.hour_bar.high_price,
                                           bar.high_price)
            self.hour_bar.low_price = min(self.hour_bar.low_price,
                                          bar.low_price)

            self.hour_bar.close_price = bar.close_price
            self.hour_bar.volume += bar.volume
            self.hour_bar.turnover += bar.turnover
            self.hour_bar.open_interest = bar.open_interest

            finished_bar = self.hour_bar
            self.hour_bar = None

        # If minute bar of new hour, then push existing window bar
        elif bar.datetime.hour != self.hour_bar.datetime.hour:
            finished_bar = self.hour_bar

            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            self.hour_bar = BarData(symbol=bar.symbol,
                                    exchange=bar.exchange,
                                    datetime=dt,
                                    gateway_name=bar.gateway_name,
                                    open_price=bar.open_price,
                                    high_price=bar.high_price,
                                    low_price=bar.low_price,
                                    close_price=bar.close_price,
                                    volume=bar.volume,
                                    turnover=bar.turnover,
                                    open_interest=bar.open_interest)
        # Otherwise only update minute bar
        else:
            self.hour_bar.high_price = max(self.hour_bar.high_price,
                                           bar.high_price)
            self.hour_bar.low_price = min(self.hour_bar.low_price,
                                          bar.low_price)

            self.hour_bar.close_price = bar.close_price
            self.hour_bar.volume += bar.volume
            self.hour_bar.turnover += bar.turnover
            self.hour_bar.open_interest = bar.open_interest

        # Push finished window bar
        if finished_bar:
            self.on_hour_bar(finished_bar)

    def on_hour_bar(self, bar: BarData) -> None:
        """"""
        if self.window == 1:
            if self.on_window_bar:
                self.on_window_bar(bar)
        else:
            if not self.window_bar:
                self.window_bar = BarData(symbol=bar.symbol,
                                          exchange=bar.exchange,
                                          datetime=bar.datetime,
                                          gateway_name=bar.gateway_name,
                                          open_price=bar.open_price,
                                          high_price=bar.high_price,
                                          low_price=bar.low_price)
            else:
                self.window_bar.high_price = max(self.window_bar.high_price,
                                                 bar.high_price)
                self.window_bar.low_price = min(self.window_bar.low_price,
                                                bar.low_price)

            self.window_bar.close_price = bar.close_price
            self.window_bar.volume += bar.volume
            self.window_bar.turnover += bar.turnover
            self.window_bar.open_interest = bar.open_interest

            self.interval_count += 1
            if not self.interval_count % self.window:
                self.interval_count = 0

                if self.on_window_bar:
                    self.on_window_bar(self.window_bar)

                self.window_bar = None

    def update_bar_daily_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create daily bar object
        if not self.daily_bar:
            self.daily_bar = BarData(symbol=bar.symbol,
                                     exchange=bar.exchange,
                                     datetime=bar.datetime,
                                     gateway_name=bar.gateway_name,
                                     open_price=bar.open_price,
                                     high_price=bar.high_price,
                                     low_price=bar.low_price)
        # Otherwise, update high/low price into daily bar
        else:
            self.daily_bar.high_price = max(self.daily_bar.high_price,
                                            bar.high_price)
            self.daily_bar.low_price = min(self.daily_bar.low_price,
                                           bar.low_price)

        # Update close price/volume/turnover into daily bar
        self.daily_bar.close_price = bar.close_price
        self.daily_bar.volume += bar.volume
        self.daily_bar.turnover += bar.turnover
        self.daily_bar.open_interest = bar.open_interest

        # Check if daily bar completed
        if bar.datetime.time() == self.daily_end:
            self.daily_bar.datetime = bar.datetime.replace(hour=0,
                                                           minute=0,
                                                           second=0,
                                                           microsecond=0)

            if self.on_window_bar:
                self.on_window_bar(self.daily_bar)

            self.daily_bar = None

    def generate(self) -> BarData:
        """
        Generate the bar data and call callback immediately.
        """
        bar: BarData | None = self.bar

        if bar:
            bar.datetime = bar.datetime.replace(second=0, microsecond=0)
            self.on_bar(bar)

        self.bar = None
        return bar
