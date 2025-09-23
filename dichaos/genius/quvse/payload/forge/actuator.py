import asyncio, os, pdb, json, hashlib, itertools
from urllib.parse import urlparse
from datetime import datetime
import pandas as pd
from joblib import Parallel, delayed
#from lumina.genetic.process import *
from dichaos.services.llm.factory import LLMServiceFactory
from utils.mongo import MongoLoader
from payload.forge.fetch import Fetch
from agents.factor.forge_agent import ForgeAgent
from payload.tool_executor import ToolExecutor
from pymongo import InsertOne, DeleteOne
from payload.forge.factors import *


def make_id(token: str) -> str:
    m = hashlib.md5()
    m.update(token.encode("utf-8"))
    return m.hexdigest()


def run_process(column, factors_data, factor_returns):
    factor_perf = FactorsPerf(init_data=False)
    status = factor_perf.calc(factor=column,
                              factors_data=factors_data,
                              factor_returns=factor_returns)
    del column['expression']
    status.update(column)
    return status


def run_perf(target_column, factors_data, factor_returns):
    try:
        perf_data = run_process(column=target_column,
                                factors_data=factors_data,
                                factor_returns=factor_returns)
    except:
        perf_data = {
            "total_ret": 0.0,
            "avg_ret": 0.0,
            "max_dd": 0.0,
            "calmar": np.nan,
            "sharpe": np.nan,
            "turnover": np.nan,
            "win_rate": np.nan,
            "profit_ratio": np.nan,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "ret_name": "nxt1_ret_1h",
            "roll_win": 60,
            "scale_method": "roll_min_max",
        }
    return perf_data


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

        self._factors_data, self._factor_returns = fetch_market(
            instruments='ims',
            method='aicso0',
            category='basic',
            name=['train', 'val', 'test'])

        self._factors_data = self._factors_data.sort_values(
            by=['trade_time', 'code']).set_index(['trade_time'])

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

    def load_data(self, table_name):
        results = self.mongo_client.find(query={},
                                         collection_name=table_name,
                                         limit=10,
                                         key_or_list={'ic_mean': -1})
        results = results.dropna(subset=['ic_ir', 'ic_mean', 'calmar'])
        return results[['expression', 'ic_mean',
                        'calmar']].to_dict(orient='records')

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

    async def interpre(self, expression, category):
        results = None
        # 使用 async with 语句来自动管理连接和断开
        async with ToolExecutor(server_url=self.url) as executor:
            # 直接调用 'factor_forge_tool'，并传递参数
            results = await executor.execute_tool(
                tool_name="factor_interpre_tool",
                expression=expression,
                category=category)
            print(results)

    async def evolve(self, category):
        ## 提取绩效最高的5个因子
        count = 5
        factors_data = self.load_data(table_name='quvse_factors_evole')
        '''
        factors_data = [{
            "expression":
            "DIV(EMA(10, SUBBED('cr003_5_10_0','tn004_5_10_0')), ABS('oi020_10_15_0'))",
            "ic": 0.02,
            "carlmar": 4
        }, {
            "expression":
            "MUL(SIGMOID(SUBBED('oi020_10_15_0', MA(10, 'cr003_5_10_0'))), 'tn004_5_10_0')",
            "ic": -0.03,
            "carlmar": 3.2
        }]
        '''
        factor_context = json.dumps(factors_data)
        async with ToolExecutor(server_url=self.url) as executor:
            # 直接调用 'factor_evolve_tool'，并传递参数
            results = await executor.execute_tool(
                tool_name="factor_evolve_tool",
                factor_context=factor_context,
                category=category,
                count=count)
        k_split = 4
        factors_infos = results['expression']['factors']
        res = Parallel(n_jobs=k_split, verbose=1)(
            delayed(run_perf)(factors_infos[i], self._factors_data,
                              self._factor_returns)
            for i in range(len(factors_infos)))
        results = pd.DataFrame(res)
        results = results.dropna(subset=['ic_ir', 'ic_mean', 'calmar'])
        self.refresh_data(
            results=results,
            table_name='quvse_factors_evole',
            keys=['expression', 'ret_name', 'roll_win', 'scale_method'])
