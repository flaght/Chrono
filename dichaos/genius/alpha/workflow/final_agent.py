import os, pdb, asyncio, time
import pandas as pd
from urllib.parse import urlparse
from pymongo import InsertOne, DeleteOne
from alphacopilot.calendars.api import advanceDateByCalendar
from agent.decision.agent import DecisionAgent
from kdutils.mongo import MongoLoader


class FinalAgent(object):

    def __init__(self, symbol, debate_rounds):
        self._decision_agent = DecisionAgent.from_config(
            path=os.path.join('agent', DecisionAgent.name))
        self._symbol = symbol
        self._debate_rounds = debate_rounds
        self._mongo_client = MongoLoader(
            connection_string=os.environ['MG_URL'],
            db_name=urlparse(os.environ['MG_URL']).path.lstrip('/'),
            collection_name='chat_history1')

    def load_battle_state(self, end_date):
        results = self._mongo_client.find(query={
            'trade_date': end_date,
            'code': self._symbol
        },
                                          collection_name='alpha_agent_debat')
        ## 转化成字典
        result_dict = results.drop(['_id', 'trade_date', 'code'],
                                   axis=1).to_dict(orient='records')
        return result_dict[0]

    async def process(self, battle_state, end_date):
        debate_transcript = []
        current_round = 0
        for event in battle_state['debate_history']:
            #if event["round"] != current_round and event["round"] == 3:
            if event["round"] in [
                    self._debate_rounds, self._debate_rounds - 1,
                    self._debate_rounds - 2
            ]:
                current_round = event["round"]
                debate_transcript.append(f"\n--- 第 {current_round} 轮 ---")
                debate_transcript.append(
                    f'{event["speaker"]}: {event["content"]}')

        full_transcript = "\n".join(debate_transcript)
        print("📜 辩论记录摘要: {0}".format(full_transcript))

        final_prediction = await self._decision_agent.agenerate_prediction(
            debate_transcript=full_transcript,
            symbol=self._symbol,
            date=end_date)
        return final_prediction

    def refresh_data(self, results, table_name, keys):
        delete_requests = [
            DeleteOne(message)
            for message in results[keys].to_dict(orient='records')
        ]

        insert_requests = [
            InsertOne(message) for message in results.to_dict(orient='records')
        ]

        requests = delete_requests + insert_requests

        self._mongo_client.bulk(requests=requests,
                                collection_name="{0}".format(table_name))

    async def run(self, end_date):
        end_date = advanceDateByCalendar(
            'china.sse', end_date, '-{0}b'.format(1)).strftime('%Y-%m-%d')
        pdb.set_trace()
        battle_state = self.load_battle_state(end_date=end_date)
        final_prediction = await self.process(battle_state=battle_state,
                                              end_date=end_date)
        results = pd.DataFrame([final_prediction.model_dump()])
        results['code'] = self._symbol
        results['trade_date'] = end_date

        self.refresh_data(results=results,
                          table_name='alpha_agent_final',
                          keys=['trade_date', 'code'])
