import os, pdb, asyncio, time
import pandas as pd
from urllib.parse import urlparse
from pymongo import InsertOne, DeleteOne
from dichaos.battle.environment import AsyncBattleEnvironment
import agent as agent_sets
from kdutils.mongo import MongoLoader
from alphacopilot.calendars.api import advanceDateByCalendar

class DebatAgent(object):

    def __init__(self, agent_names, model_date, symbol, debate_rounds):
        self._agent_names = agent_names
        self._symbol = symbol
        self._environment = AsyncBattleEnvironment(debate_rounds=debate_rounds)
        self._agents = [
            getattr(agent_sets, '{0}Predict'.format(agent))(
                date=model_date,
                config_path=os.path.join("agent"),
                memory_path=os.path.join("records"),
                symbol=symbol).agent for agent in agent_names
        ]

        self._mongo_client = MongoLoader(
            connection_string=os.environ['MG_URL'],
            db_name=urlparse(os.environ['MG_URL']).path.lstrip('/'),
            collection_name='chat_history1')

    def load_agent_analysis(self, end_date):
        results = self._mongo_client.find(
            query={
                'trade_date': end_date,
                'code': self._symbol
            },
            collection_name='alpha_agent_analysis')
        ## 转化成字典
        result_dict = results.drop(
            ['_id', 'trade_date', 'code'],
            axis=1).set_index('name')['decision'].to_dict()
        return result_dict

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

    async def procees(self, initial_reports_map):
        for agent_instance in self._agents:
            agent_name = agent_instance.name
            role = agent_instance.desc()
            initial_analysis_for_agent = initial_reports_map.get(
                agent_name, "我没有生成初始分析。")
            self._environment.register_agent(
                llm_provider=agent_instance.llm_provider,
                agent=agent_instance,
                role_description=role,
                initial_analysis=initial_analysis_for_agent)
        # 将所有报告汇总，作为辩论的全局上下文
        report_str = "\n".join([
            f"- **{report_name}**: {report_text}"
            for report_name, report_text in initial_reports_map.items()
        ])

        research_report = {
            "symbol": self._symbol,
            "preliminary_analysis_summary": report_str
        }

        final_results = await self._environment.run(research_report) 
        return final_results

    async def run(self, end_date):
        end_date = advanceDateByCalendar(
            'china.sse', end_date, '-{0}b'.format(1)).strftime('%Y-%m-%d')
        initial_reports_map = self.load_agent_analysis(end_date=end_date)
        final_results = await self.procees(initial_reports_map=initial_reports_map)
        final_results = pd.DataFrame([final_results])
        final_results['trade_date'] = end_date
        final_results['code'] = self._symbol
        self.refresh_data(results=final_results,
                          table_name='alpha_agent_debat',
                          keys=['code', 'trade_date'])
