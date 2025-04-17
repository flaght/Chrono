import os
from kdutil.mongodb import MongoDBManager
from macro.contract import MAIN_CONTRACT_MAPPING
from .factorx import Factorx


class Engine(object):

    def __init__(self, code):
        self._mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
        self.factorx = Factorx(symbol=MAIN_CONTRACT_MAPPING[code], n_job=4)

    def run(self, trade_time):
        _ = self.factorx.impluse_run(trade_time=trade_time)
