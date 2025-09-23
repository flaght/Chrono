import os, pdb, asyncio, time
import pandas as pd
from urllib.parse import urlparse
import agent as agent_sets
from kdutils.mongo import MongoLoader
from pymongo import InsertOne, DeleteOne

from alphacopilot.calendars.api import advanceDateByCalendar


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

    async def process(self, begin_date, end_date, concurrent_num):
        for p in self._predictors:
            p.prepare_data(begin_date=begin_date, end_date=end_date)

        results = await self.concurrent_predictions(
            end_date=end_date, concurrent_num=concurrent_num)

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

    async def run(self, end_date, concurrent_num=2):
        end_date = advanceDateByCalendar(
            'china.sse', end_date, '-{0}b'.format(1)).strftime('%Y-%m-%d')

        initial_analysis_map, reason_reports_map = await self.process(
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
                          table_name='alpha_agent_analysis',
                          keys=['name', 'code', 'trade_date'])

        self.refresh_data(results=pd.DataFrame(reason_reports),
                          table_name='alpha_agent_reason',
                          keys=['name', 'code', 'trade_date'])
