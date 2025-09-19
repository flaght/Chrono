import asyncio, os, pdb, json, hashlib
from urllib.parse import urlparse
from datetime import datetime
import pandas as pd
from dichaos.services.llm.factory import LLMServiceFactory
from utils.mongo import MongoLoader
from payload.forge.fetch import Fetch
from agents.factor.forge_agent import ForgeAgent
from payload.tool_executor import ToolExecutor
from pymongo import InsertOne, DeleteOne

def make_id(token: str) -> str:
    m = hashlib.md5()
    m.update(token.encode("utf-8"))
    return m.hexdigest()


class Actuator(object):

    def __init__(self, url, base_path):
        self.url = url
        self.base_path = base_path
        self.fetch = Fetch()
        self.llm_service = LLMServiceFactory.create_llm_service(
            llm_model=os.environ['MODEL_NAME'],
            llm_provider=os.environ['MODEL_PROVIDER'],
            system_message='')
        self.mongo_client = MongoLoader(
            connection_string=os.environ['MG_URL'],
            db_name=urlparse(os.environ['MG_URL']).path.lstrip('/'),
            collection_name='chat_history1')

    def fetch_data(self, path_file, k=70):
        return self.fetch.fetch_random_features(path_file=path_file, k=k)

    def refresh_data(self, results, table_name, keys):
        delete_requests = [
            DeleteOne(message)
            for message in results[keys].to_dict(orient='records')
        ]

        insert_requests = [
            InsertOne(message) for message in results.to_dict(orient='records')
        ]

        requests = delete_requests + insert_requests

        self.mongo_client.bulk(requests=requests,
                               collection_name="{0}".format(table_name))

    async def create(self, agent, user_request, **kwargs):

        def make_id(token: str) -> str:
            m = hashlib.md5()
            m.update(token.encode("utf-8"))
            return m.hexdigest()

        await agent.connect_tool_server()
        results = await agent.run(request=user_request)
        factors_data = json.loads(results)['expression']['factors']
        factors_data = pd.DataFrame(factors_data)
        factors_data['name'] = factors_data["expression"].apply(make_id)
        return factors_data

    async def run(self, category):
        path_file = os.path.join(self.base_path,
                                 "{}_fields_dependencies.csv".format(category))
        data = self.fetch_data(path_file)
        agent_kwargs = {
            "llm_service": self.llm_service,
            "generate_final_report": False
        }

        agent_kwargs["tool_server_config"] = {
            'transport': 'http',
            'base_url': self.url
        }
        user_request = json.dumps(data, ensure_ascii=False)
        agent = await ForgeAgent.create(**agent_kwargs)
        factors_data = await self.create(agent=agent,
                                         user_request=user_request)
        factors_data['model'] = os.environ['MODEL_NAME']
        factors_data['provider'] = os.environ['MODEL_PROVIDER']
        factors_data['timestampe'] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        factors_data['category'] = category
        self.refresh_data(results=factors_data,
                          table_name='quvse_factors_details',
                          keys=['name'])

    async def run1(self, category):
        # 2. 使用 ToolExecutor 直接调用远程工具
        results = None
        # 使用 async with 语句来自动管理连接和断开
        async with ToolExecutor(server_url=self.url) as executor:
            # 直接调用 'factor_forge_tool'，并传递参数
            results = await executor.execute_tool(
                tool_name="factor_forge_tool", count=10, category='basic')

        # 3. 处理返回结果
        if not results or 'expression' not in results or 'factors' not in results[
                'expression']:
            print("❌ [Actuator] 工具返回的结果格式不符合预期。")
            return
        factors_data = pd.DataFrame(results['expression']['factors'])
        factors_data['name'] = factors_data['expression'].apply(make_id)

        # 4. 存储数据
        factors_data['model'] = os.environ['MODEL_NAME']
        factors_data['provider'] = os.environ['MODEL_PROVIDER']
        factors_data['timestampe'] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        factors_data['category'] = category
        self.refresh_data(results=factors_data,
                          table_name='quvse_factors_details',
                          keys=['name'])

        print(f"✅ [Actuator] 成功生成并存储了 {len(factors_data)} 条策略。")
