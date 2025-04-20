import os, importlib, datetime, pdb
from pymongo import InsertOne, DeleteOne
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import SubscribeRequest
from vnpy.trader.event import EVENT_LOG, EVENT_CONTRACT, EVENT_TICK, EVENT_ACCOUNT
from kdutil.mongodb import MongoDBManager
from const import STATE, CacheBar, ContractTuple, BarData
from macro.contract import *


class Conor(object):

    def __init__(self, qubit, code, name='ctp'):
        self.name = name
        self._mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
        self.event_engine = EventEngine()
        self.event_engine.register(EVENT_LOG, self.process_log_event)
        self.event_engine.register(EVENT_ACCOUNT, self.process_account)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract)
        self.event_engine.register(EVENT_TICK, self.process_tick)

        self.main_engine = MainEngine(self.event_engine)
        module_name = "vnpy_{0}".format(self.name)
        try:
            module_class = importlib.import_module(module_name)
            getway_name = "{0}Gateway".format(self.name.capitalize())
            getway_module = module_class.__getattribute__(getway_name)
        except ImportError as e:
            raise (str(e))
        self.main_engine.add_gateway(getway_module)
        self._state_list = {STATE.INIT}  ##  状态机
        self.code = code
        self._subscribe = [MAIN_CONTRACT_MAPPING[code]]
        self.contracts = {}
        self.bars = {}
        for symbol in self._subscribe:
            self.bars[symbol] = CacheBar(symbol=symbol)
        self.qubit = qubit

    def update_bar(self, data, table_name):
        insert_request = [
            InsertOne(data)  # for data in data.to_dict(orient='records')
        ]

        delete_request = [
            DeleteOne({
                'symbol': data['symbol'],
                'exchange': data['exchange'],
                'datetime': data['datetime']
            })
        ]
        _ = self._mongo_client['neutron'][table_name].bulk_write(
            delete_request + insert_request, bypass_document_validation=True)

    def process_log_event(self, event):
        log = event.data
        print(f"{log.time}\t{log.msg}")

    def process_contract(self, event):
        if STATE.CONTRACT not in self._state_list:
            self._state_list.add(STATE.CONTRACT)

    def process_account(self, event):
        if STATE.LOGIN_ON not in self._state_list:
            self._state_list.add(STATE.LOGIN_ON)
            print("account {0} {1} login on".format(event.data.gateway_name,
                                                    event.data.accountid))
            for symbol in self._subscribe:
                contract = self.contracts[symbol]
                req = SubscribeRequest(symbol=contract.symbol,
                                       exchange=contract.exchange)
                #print(req)
                self.main_engine.subscribe(req, contract.getway_name)

    def process_contract(self, event):
        if STATE.CONTRACT not in self._state_list:
            self._state_list.add(STATE.CONTRACT)

        self.contracts[event.data.symbol] = ContractTuple(
            getway_name=event.data.gateway_name,
            symbol=event.data.symbol,
            exchange=event.data.exchange,
            name=event.data.name)

    def process_tick(self, event):
        if STATE.MARKET_TICK not in self._state_list:
            self._state_list.add(STATE.MARKET_TICK)
        data = event.data.__dict__

        data['exchange'] = data['exchange'].value
        data['datetime'] = data['datetime'].strftime('%Y-%m-%d %H:%M:%S')
        insert_request = [InsertOne(data)]

        cache_bar = self.bars[event.data.symbol]

        tick = event.data
        tick_time = datetime.datetime.strptime(tick.datetime,
                                               '%Y-%m-%d %H:%M:%S')
        tickMinute = tick_time.minute
        if (tickMinute != cache_bar.minute):  # or tickMinute == 0:
            if cache_bar.bar:
                ## 存储
                print(
                    "tickMinute:{0} cache_bar.minute:{1} tickMinute:{2} bar:{3}"
                    .format(tickMinute, cache_bar.minute, tickMinute,
                            cache_bar.bar.__dict__))
                data = cache_bar.bar.__dict__
                print(data)
                ## vwap
                current_time = datetime.datetime.strptime(
                    cache_bar.bar.datetime, '%Y-%m-%d %H:%M:%S')
                #_ = self._mongo_client['neutron']['market_bar'].bulk_write(
                ##    insert_request, bypass_document_validation=True)
                data['vwap'] = data['value'] / data['volume'] / int(
                    CONT_MULTNUM_MAPPING[
                        self.code]) if data['volume'] != 0.0 else 0
                #self.update_bar(data=data, table_name='market_bar')
                self.qubit.run(trade_time=current_time)
            bar = BarData()
            bar.vt_symbol = tick.vt_symbol
            bar.symbol = tick.symbol
            bar.exchange = tick.exchange
            bar.open = tick.last_price
            bar.high = tick.last_price
            bar.low = tick.last_price
            bar.close = tick.last_price
            bar.date = tick_time.date().strftime('%Y-%m-%d')
            bar.time = tick_time.time().strftime('%H:%M:%S')
            bar.datetime = tick.datetime
            bar.volume = tick.volume
            bar.value = tick.turnover
            bar.value1 = tick.last_price * tick.volume
            bar.open_interest = tick.open_interest

            cache_bar.bar = bar
            cache_bar.minute = tickMinute
        else:
            bar = cache_bar.bar
            bar.high = max(bar.high, tick.last_price)
            bar.low = min(bar.low, tick.last_price)
            bar.close = tick.last_price
            bar.volume += tick.volume
            bar.value += tick.turnover
            bar.value1 += tick.last_price * tick.volume
            bar.open_interest = tick.open_interest

        self.bars[event.data.symbol] = cache_bar

    def contract(self, symbol):
        self.main_engine.get_contract(symbol)

    def create_config(self, **kwargs):
        config = {}
        config["用户名"] = kwargs['account_id']
        config["密码"] = kwargs['password']
        config["经纪商代码"] = kwargs['broker_id']
        config["交易服务器"] = kwargs['td_address']
        config["行情服务器"] = kwargs['md_address']
        config["产品名称"] = kwargs['app_id']
        config["授权编码"] = kwargs['auth_code']
        return config

    def start(self, **kwargs):
        if 'account_id' in kwargs:
            self._account_id = kwargs['account_id']
        config = self.create_config(**kwargs)
        self.main_engine.connect(config, self.name.upper())
