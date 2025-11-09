import os, pdb, asyncio, time
import pandas as pd
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar
from dichaos.battle.environment import AsyncBattleEnvironment
from dichaos.battle.state import BattleState
from agent.decision.agent import DecisionAgent
from kdutils.report import ReportGenerator
import agent as agent_sets
from kdutils.mongo import MongoLoader
from pymongo import InsertOne, DeleteOne


class PredictAgent(object):

    def __init__(self, agent_names, model_date, symbol):
        self._predictors = [
            getattr(agent_sets, '{0}Predict'.format(agent))(
                date=model_date,
                config_path=os.path.join("agent"),
                memory_path=os.path.join("records"),
                symbol=symbol) for agent in agent_names
        ]
        self._symbol = symbol
        self._mongo_client = MongoLoader(
            connection_string=os.environ['MG_URL'],
            db_name=urlparse(os.environ['MG_URL']).path.lstrip('/'),
            collection_name='chat_history1')

    async def predict_with_semaphore(self, predictor,
                                     semaphore: asyncio.Semaphore, date: str):
        """
        一个异步包装函数，它在使用 predictor 进行预测之前，会先从 semaphore 获取许可。
    
        Args:
            predictor: 一个 Predictor 类的实例 (例如 CloutoPredict)。
            semaphore (asyncio.Semaphore): 用于控制并发的信号量。
            date (str): 预测日期。

        Returns:
            The result of the prediction or the exception if it fails.
        """
        # 3. 在包装协程内获取 Semaphore
        # async with 语句确保了即使在预测过程中发生异常，信号量也总能被正确释放。
        async with semaphore:
            # 当协程执行到这里时，它已经成功获取了一个“令牌”。
            # 如果令牌已满，它会在上一行异步地等待。
            print(f"[{predictor.agent.name}] 获取到信号量许可，开始执行预测...")

            try:
                # 执行实际的预测调用
                result = await predictor.agenerate_prediction(
                    date=date, predict_data=predictor.create_data(date=date))
                return result
            except Exception as e:
                # 捕获并返回异常，这样 gather 就不会中断
                print(f"[{predictor.agent.name}] 预测时发生错误: {e}")
                return e

    async def concurrent_predictions(self, end_date: str, concurrent_num: int):
        semaphore = asyncio.Semaphore(concurrent_num)
        # 创建一个任务列表，这次调用的是我们的包装函数 predict_with_semaphore
        tasks = [
            self.predict_with_semaphore(p, semaphore, end_date)
            for p in self._predictors
        ]
        print("已创建所有并发任务，准备使用 asyncio.gather 运行...")
        results = await asyncio.gather(*tasks)
        return results

    def process(self, begin_date, end_date, concurrent_num):
        for p in self._predictors:
            p.prepare_data(begin_date=begin_date, end_date=end_date)

        results = asyncio.run(
            self.concurrent_predictions(end_date=end_date,
                                        concurrent_num=concurrent_num))

        # 处理并收集结果
        analysis_reports = {}  # key 是 agent.name, value 是分析报告字符串
        reason_reports = {}
        for pred, result in zip(self._predictors, results):
            agent_name = pred.agent.name

            summary = getattr(result, 'summary', str(result.summary))

            reason = getattr(result, 'reasoning', str(result.reasoning))

            analysis_details = getattr(result, 'analysis_details',
                                       str(result.analysis_details))

            analysis_reports[agent_name] = "{0} {1}".format(
                analysis_details, summary)

            reason_reports[agent_name] = reason
        return analysis_reports, reason_reports

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

    def run(self, end_date, concurrent_num=2):
        end_date = advanceDateByCalendar(
            'china.sse', end_date, '-{0}b'.format(1)).strftime('%Y-%m-%d')
        pdb.set_trace()
        initial_analysis_map, reason_reports_map = self.process(
            begin_date=end_date,
            end_date=end_date,
            concurrent_num=concurrent_num)

        initial_analysis = [{
            'trade_date': end_date,
            'code': self._symbol,
            'name': key,
            'decision': value
        } for key, value in initial_analysis_map.items()]

        reason_reports = [{
            'trade_date': end_date,
            'code': self._symbol,
            'name': key,
            'reasoning': value
        } for key, value in reason_reports_map.items()]

        self.refresh_data(results=pd.DataFrame(initial_analysis),
                          table_name='compass_agent_analysis',
                          keys=['name', 'code', 'trade_date'])

        self.refresh_data(results=pd.DataFrame(reason_reports),
                          table_name='compass_agent_reason',
                          keys=['name', 'code', 'trade_date'])


class DebatAgent(object):

    def __init__(self, agent_names, model_date, symbol, debate_rounds):
        self._agent_names = agent_names
        self._symbol = symbol
        self._environment = AsyncBattleEnvironment(debate_rounds=debate_rounds)
        self._agents = [
            getattr(agent_sets, '{0}Predict'.format(agent))(
                date=model_date,
                memory_path=os.path.join("records"),
                config_path=os.path.join("agent"),
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
            collection_name='compass_agent_analysis')
        ## 转化成字典
        pdb.set_trace()
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

    def procees(self, initial_reports_map):
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

        final_results = asyncio.run(self._environment.run(research_report))
        return final_results

    def run(self, end_date):
        end_date = advanceDateByCalendar(
            'china.sse', end_date, '-{0}b'.format(1)).strftime('%Y-%m-%d')
        initial_reports_map = self.load_agent_analysis(end_date=end_date)
        final_results = self.procees(initial_reports_map=initial_reports_map)
        final_results = pd.DataFrame([final_results])
        final_results['trade_date'] = end_date
        final_results['code'] = self._symbol
        pdb.set_trace()
        self.refresh_data(results=final_results,
                          table_name='compass_agent_debat',
                          keys=['code', 'trade_date'])


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
        results = self._mongo_client.find(
            query={
                'trade_date': end_date,
                'code': self._symbol
            },
            collection_name='compass_agent_debat')
        ## 转化成字典

        result_dict = results.drop(['_id', 'trade_date', 'code'],
                                   axis=1).to_dict(orient='records')
        return result_dict[0]

    def process(self, battle_state, end_date):
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

        final_prediction = asyncio.run(
            self._decision_agent.agenerate_prediction(
                debate_transcript=full_transcript,
                symbol=self._symbol,
                date=end_date))
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

    def run(self, end_date):
        end_date = advanceDateByCalendar(
            'china.sse', end_date, '-{0}b'.format(1)).strftime('%Y-%m-%d')
        battle_state = self.load_battle_state(end_date=end_date)
        final_prediction = self.process(battle_state=battle_state,
                                        end_date=end_date)
        results = pd.DataFrame([final_prediction.model_dump()])
        results['code'] = self._symbol
        results['trade_date'] = end_date

        self.refresh_data(results=results,
                          table_name='compass_agent_final',
                          keys=['trade_date', 'code'])


class Report(object):

    def __init__(self, symbol, base_path, resource_path):
        self._base_path = base_path
        self._resource_path = resource_path
        self._symbol = symbol
        self._mongo_client = MongoLoader(
            connection_string=os.environ['MG_URL'],
            db_name=urlparse(os.environ['MG_URL']).path.lstrip('/'),
            collection_name='chat_history1')

    def load_data(self, end_date, table_name):
        results = self._mongo_client.find(query={
            'trade_date': end_date,
            'code': self._symbol
        },
                                          collection_name=table_name)
        return results

    def process(self, end_date):
        final_prediction = self.load_data(end_date=end_date,
                                          table_name='compass_agent_final')
        final_prediction = final_prediction.drop(['_id'],
                                                 axis=1).loc[0].to_dict()
        final_battle_state = self.load_data(end_date=end_date,
                                            table_name='compass_agent_debat')
        final_battle_state = final_battle_state.drop(['_id'],
                                                     axis=1).loc[0].to_dict()
        reason_reports = self.load_data(end_date=end_date,
                                        table_name='compass_agent_reason')
        reason_reports = reason_reports[[
            'name', 'reasoning'
        ]].set_index('name')['reasoning'].to_dict()
        report_data = {
            "date": end_date,
            "symbol": self._symbol,
            "final_prediction": final_prediction,
            "battle_results": final_battle_state,
            "reason_reports": reason_reports
        }
        return report_data

    def run(self, end_date):
        end_date1 = advanceDateByCalendar(
            'china.sse', end_date, '-{0}b'.format(1)).strftime('%Y-%m-%d')
        report = ReportGenerator(
            output_dir=os.path.join(self._base_path, end_date, "report",
                                    "html"),
            template_name=os.path.join(self._resource_path,
                                       "report_template.html"))
        report_data = self.process(end_date=end_date1)
        report_data['date'] = end_date
        report.run(report_data=report_data)


model_date = '2022-09-05'
trade_date = '2025-11-07'  ##预测日期
debate_rounds = 4
### 并发预测


agents = PredictAgent(
    agent_names=['Clouto', 'Indicator', 'MoneyFlow', 'PosFlow', 'Chip'],
    model_date=model_date,
    symbol='IM')

agents.run(end_date=trade_date) ## 内部处理了时间偏移-1


agents = DebatAgent(
    agent_names=['Clouto', 'Indicator', 'MoneyFlow', 'PosFlow', 'Chip'],
    #agent_names=['Indicator'],
    model_date=model_date,
    symbol='IM',
    debate_rounds=debate_rounds)
agents.run(end_date=trade_date)



agents = FinalAgent(symbol='IM', debate_rounds=debate_rounds)
agents.run(end_date=trade_date)




base_path = "/workspace/worker/temp/nginx/opts/dichaos/futures"
resource_path = "resource"
report = Report(symbol='IM', base_path=base_path, resource_path=resource_path)
report.run(end_date=trade_date)