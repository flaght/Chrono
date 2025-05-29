from typing import Dict
from pydantic import BaseModel
import pdb, json, time
import pandas as pd
from dichaos.kdutils.logger import logger
from dichaos.agents.agents import Agents
from .model import KLine, Risker


class Agent(Agents):
    name = 'risker'
    category = 'indexor'

    def __init__(self, name: str, top_k: int, vector_provider: str,
                 db_name: str, embedding_model: str, embedding_provider: str,
                 llm_model: str, llm_provider: str, memory_params: Dict):
        super(Agent, self).__init__(name=name,
                                    top_k=top_k,
                                    vector_provider=vector_provider,
                                    db_name=db_name,
                                    embedding_model=embedding_model,
                                    embedding_provider=embedding_provider,
                                    llm_model=llm_model,
                                    llm_provider=llm_provider,
                                    memory_params=memory_params,
                                    system_message="")

    def create_kline(self, date, symbol, open, close, high, low, volume):
        return KLine(date=date,
                     symbol=symbol,
                     open=open,
                     close=close,
                     high=high,
                     low=low,
                     volume=volume)

    def handing_data(self, trade_date: str, symbol: str, kline: BaseModel):

        self.brain_db.add_memory_short_term(symbol=symbol,
                                            date=trade_date,
                                            text=kline.model_dump_json())

    def query_records(self, trade_date: str, symbol: str):
        short_records = self.brain_db.query_memory_short_term(
            query_text="date is {0}".format(trade_date),
            symbol=symbol,
            top_k=self.top_k * 1000,
            duplicates=True)
        for index, record in zip(short_records[1], short_records[0]):
            memory = json.loads(record)
            memory = KLine(**memory)
        return memory

    def generate(self, date: str, symbol: str, kline: BaseModel,
                 portfolio: Dict):
        pdb.set_trace()
        current_price = kline.close
        current_position_value = portfolio.get("cost_basis", 0)
        total_portfolio_value = portfolio.get("cash", 0) + portfolio.get(
            "cost_basis", {})
        position_limit = total_portfolio_value * 0.20

        remaining_position_limit = position_limit - current_position_value

        max_position_size = min(remaining_position_limit,
                                portfolio.get("cash", 0))

        return Risker(date=date,
                      remaining_position_limit=remaining_position_limit,
                      current_price=current_price,
                      portfolio_value=total_portfolio_value,
                      current_position=current_position_value,
                      position_limit=position_limit,
                      remaining_limit=max_position_size,
                      available_cash=portfolio.get("cash", 0))
