import asyncio, pdb, os
import pandas as pd
from html import escape
from kdutils.macro import base_path
from lib.agt001 import create_agent
from features.text.util import neutralize_prompt_braces
from features.text.mode import NewsEventResult, TextualFeatureResult
from lib.inkits.utils import build_dynamic_schema, get_fewshot_template2


async def generate_with_semaphore(agent_instance,
                                  trade_time,
                                  semaphore,
                                  human_message,
                                  params_dict,
                                  schema_cls,
                                  key_name='events'):
    async with semaphore:
        print(f"[{agent_instance.name}] 获取到并发许可，开始极速发散推演...")
        try:
            result = await agent_instance.agenerate_message(
                human_message=human_message,
                params=params_dict,
                response_schema=schema_cls)

            output = result.model_dump()
            for event in output[key_name]:
                # 对下游只暴露统一字段，消除模型输出形状差异。
                event.pop('title', None)
                event.pop('description', None)
            output['name'] = agent_instance.name
            output['trade_time'] = trade_time
            return {"output": output, "status": 0}
        except Exception as e:
            print(f"❌ [{agent_instance.name}] 生成期间发生错误: {e}")
            return {"output": "", "status": -1}


class AgentFeature(object):

    def __init__(self, max_count=1):
        self.semaphore = asyncio.Semaphore(max_count)

    ### news 新闻
    def _map_event(self, trade_date, group1, agent, thought, max_events,
                   max_content_chars):
        tasks = []
        for batch_index, batch_rows in enumerate(group1, start=1):
            items = []
            for row in batch_rows:
                publish_time = pd.to_datetime(row['publish_time']).isoformat()
                content = row['content'].strip()[:max_content_chars]
                item = (
                    "<item>\n"
                    #f"  <source_id>{escape(row['source_id'])}</source_id>\n"
                    f"  <publish_time>{escape(publish_time)}</publish_time>\n"
                    f"  <content>{escape(content)}</content>\n"
                    "</item>")
                items.append(item)

            news_str = "\n\n".join(items)
            batch_id = f"B{batch_index:02d}"
            data_cutoff_time = f"{trade_date}T23:59:59+08:00"
            min_importance = 3
            json_structure = get_fewshot_template2(NewsEventResult)
            params = {
                "trade_date": trade_date,
                "batch_id": batch_id,
                "data_cutoff_time": data_cutoff_time,
                "min_importance": min_importance,
                "max_events": int(max_events),
                # 新闻正文可能包含 JSON/代码；使用字符实体规避多层模板解析。
                "news_items": neutralize_prompt_braces(news_str),
                "json_structure": get_fewshot_template2(NewsEventResult)
            }
            thougths_cot = thought["cot"]
            # prompt = ("任务参数：\n"
            #           "- trade_date：{trade_date}\n"
            #           "- batch_id：{batch_id}\n"
            #           "- data_cutoff_time：{data_cutoff_time}\n"
            #           "- min_importance：{min_importance}\n"
            #           "- max_events：{max_events}\n\n"
            #           "\n\n输出格式必须为:\n + {{json_structure}}"
            #           )
            prompt = f"""
                - trade_date：{trade_date}\n
                - batch_id：{batch_id}\n
                - data_cutoff_time：{data_cutoff_time}\n
                - min_importance：{min_importance}\n
                - max_events：{max_events}\n\n
                {thougths_cot}
                \n\n输出格式必须为:\n
                {{json_structure}}
                """
            tasks.append(
                generate_with_semaphore(trade_time=trade_date,
                                        agent_instance=agent,
                                        semaphore=self.semaphore,
                                        human_message=prompt,
                                        params_dict=params,
                                        schema_cls=NewsEventResult))
        return tasks

    async def create_map_event(self,
                               news_data,
                               max_events=10,
                               batch_size=40,
                               max_content_chars=800,
                               is_refresh=False):
        base_dirs = os.path.join(base_path, "data", "event", "map")
        os.makedirs(base_dirs, exist_ok=True)
        llm_name = 'deepseek_4001'  ## 指定大模型 包括地址 参数 都存储在对应字典
        vector_name = 'embedding_10002'  ##  指定嵌入模型  包括地址 参数 都存储在对应字典
        persona_name = 'news_event_map_system_100001'
        thoughts_name = 'news_event_map_user_100001'
        agent_name = 'event_map'
        agent_title = 'event_map'
        agent, thoughts1 = await create_agent(llm_name=llm_name,
                                              vector_name=vector_name,
                                              persona_name=persona_name,
                                              thoughts_name=thoughts_name,
                                              agent_name=agent_name,
                                              agent_title=agent_title,
                                              category="diver")

        pdb.set_trace()
        # news_data = news_data[(news_data['date']>='2025-08-02')&(news_data['date']<='2025-08-20')]
        for k, v in news_data.groupby('date'):
            trade_date = k
            filename = os.path.join(base_dirs,
                                    "{0}.feather".format(trade_date))
            if (os.path.exists(filename) and not is_refresh):
                print("{0}  already exists".format(filename))
                continue
            v = v.sort_values('publish_time').reset_index(drop=True)

            v1 = v.to_dict(orient='records')
            ### 切割
            group1 = [
                v1[i:i + batch_size] for i in range(0, len(v1), batch_size)
            ]
            tasks = self._map_event(trade_date=k,
                                    group1=group1,
                                    agent=agent,
                                    thought=thoughts1[thoughts_name],
                                    max_events=max_events,
                                    max_content_chars=max_content_chars)
            # 必须等当天所有批次任务创建完成后再统一并发执行一次。
            batch_results = await asyncio.gather(*tasks)
            daily_events = [
                event for result in batch_results if result['status'] == 0
                for event in result['output']['events']
            ]
            events_data = pd.DataFrame(daily_events)

            ## 暂时存本地, 用个钩子函数，把结果处理钩进来
            events_data.to_feather(filename)

    async def create_reduce_event(self,
                                  event_data,
                                  max_events=10,
                                  min_importance=3,
                                  max_content_chars=800,
                                  is_refresh=False):

        async def run_task(tasks):
            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                trade_date = result['output']['trade_time']
                event_data = pd.DataFrame(result['output']['events'])
                filename = os.path.join(base_dirs,
                                        "{0}.feather".format(trade_date))
                event_data.to_feather(filename)

        base_dirs = os.path.join(base_path, "data", "event", "reduce")
        os.makedirs(base_dirs, exist_ok=True)
        llm_name = 'deepseek_4001'  ## 指定大模型 包括地址 参数 都存储在对应字典
        vector_name = 'embedding_10002'  ##  指定嵌入模型  包括地址 参数 都存储在对应字典
        persona_name = 'news_event_reduce_system_100001'
        thoughts_name = 'news_event_reduce_user_100001'
        agent_name = 'reduce_map'
        agent_title = 'reduce_map'

        agent, thoughts1 = await create_agent(llm_name=llm_name,
                                              vector_name=vector_name,
                                              persona_name=persona_name,
                                              thoughts_name=thoughts_name,
                                              agent_name=agent_name,
                                              agent_title=agent_title,
                                              category="diver")
        thought = thoughts1[thoughts_name]
        tasks = []
        ### 10组执行一次
        for k, v in event_data.groupby('date'):
            trade_date = k
            print(trade_date)
            filename = os.path.join(base_dirs,
                                    "{0}.feather".format(trade_date))
            if (os.path.exists(filename) and not is_refresh):
                continue
            v1 = v.to_dict(orient='records')
            items = []
            for row in v1:
                content = row['summary'].strip()[:max_content_chars]
                item = ("<item>\n"
                        f"  <event_id>{escape(row['event_id'])}</event_id>\n"
                        f"  <status>{(row['status'])}</status>\n"
                        f"  <importance>{(row['importance'])}</importance>\n"
                        f"  <category>{escape((row['category']))}</category>\n"
                        f"  <content>{escape(content)}</content>\n"
                        "</item>")
                items.append(item)
            thougths_cot = thought["cot"]
            event_str = "\n\n".join(items)
            json_structure = get_fewshot_template2(NewsEventResult)
            params = {
                "trade_date": k,
                "min_importance": min_importance,
                "max_events": int(max_events),
                # 新闻正文可能包含 JSON/代码；使用字符实体规避多层模板解析。
                "candidate_events": neutralize_prompt_braces(event_str),
                "json_structure": json_structure
            }
            #prompt = thought["cot"]
            prompt = f"""
            {thougths_cot}
            \n\n输出格式必须为:\n
                {{json_structure}}
            """
            tasks.append(
                generate_with_semaphore(
                    trade_time=k,
                    agent_instance=agent,
                    semaphore=self.semaphore,
                    human_message=prompt,
                    params_dict=params,
                    schema_cls=NewsEventResult,
                ))
            if len(tasks) >= 1:
                await run_task(tasks)
                tasks = []

        if len(tasks) > 0:
            await run_task(tasks)

    async def create_event_feature(self,
                                   gov_data,
                                   cctv_data,
                                   monetary_data,
                                   event_data,
                                   dates,
                                   is_refresh=False):

        async def run_task(tasks):
            batch_results = await asyncio.gather(*tasks)
            for result in batch_results:
                trade_date = result['output']['trade_time']
                textuals_data = pd.DataFrame(result['output']['textuals'])
                filename = os.path.join(base_dirs,
                                        "{0}.feather".format(trade_date))
                textuals_data.to_feather(filename)

        base_dirs = os.path.join(base_path, "data", "event", "textuals")
        os.makedirs(base_dirs, exist_ok=True)
        llm_name = 'deepseek_4001'  ## 指定大模型 包括地址 参数 都存储在对应字典
        vector_name = 'embedding_10002'  ##  指定嵌入模型  包括地址 参数 都存储在对应字典
        persona_name = 'news_event_system_100001'
        thoughts_name = 'news_event_user_100001'
        agent_name = 'event'
        agent_title = 'event'

        agent, thoughts1 = await create_agent(llm_name=llm_name,
                                              vector_name=vector_name,
                                              persona_name=persona_name,
                                              thoughts_name=thoughts_name,
                                              agent_name=agent_name,
                                              agent_title=agent_title,
                                              category="diver")

        min_importance = 3
        max_features = 6
        tasks = []
        for dt in dates:
            ### gov
            trade_date = dt
            filename = os.path.join(base_dirs,
                                    "{0}.feather".format(trade_date))
            if (os.path.exists(filename) and not is_refresh):
                continue
            sub_gov = gov_data[gov_data['date'] == dt].to_dict(
                orient='records')
            sub_cctv = cctv_data[cctv_data['date'] == dt].to_dict(
                orient='records')
            sub_monetary = monetary_data[monetary_data['date'] == dt].to_dict(
                orient='records')
            sub_event = event_data[event_data['date'] == dt].to_dict(
                orient='records')

            gov_iems = []
            for sg in sub_gov:
                item = (
                    "<item>\n"
                    f"  <publish_time>{escape(sg['publish_time'].strftime('%Y-%m-%d %H:%M:%S'))}</publish_time>\n"
                    f"  <content>{escape(sg['content'])}</content>\n"
                    "</item>")
                gov_iems.append(item)

            cctv_iems = []
            for sg in sub_cctv:
                publish_time = sg['date'] if pd.isna(sg['publish_time']) else sg['publish_time'].strftime('%Y-%m-%d %H:%M:%S')
                item = (
                    "<item>\n"
                    f"  <publish_time>{escape(publish_time)}</publish_time>\n"
                    f"  <content>{escape(sg['content'])}</content>\n"
                    "</item>")
                cctv_iems.append(item)

            monetary_iems = []
            for sg in sub_monetary:
                publish_time = sg['date'] if pd.isna(sg['publish_time']) else sg['publish_time'].strftime('%Y-%m-%d %H:%M:%S')
                item = (
                    "<item>\n"
                    f"  <publish_time>{escape(publish_time)}</publish_time>\n"
                    f"  <content>{escape(sg['content'])}</content>\n"
                    "</item>")
                monetary_iems.append(item)

            event_iems = []
            for sg in sub_event:
                item = ("<item>\n"
                        f"  <event_time>{escape(sg['date'])}</event_time>\n"
                        f"  <content>{escape(sg['summary'])}</content>\n"
                        f"  <status>{escape(sg['status'])}</status>\n"
                        f"  <category>{escape(sg['category'])}</category>\n"
                        "</item>")
                event_iems.append(item)

            gov_str = "\n\n".join(gov_iems)
            cctv_str = "\n\n".join(cctv_iems)
            monetary_str = "\n\n".join(monetary_iems)
            event_str = "\n\n".join(event_iems)

            params = {
                "trade_date": dt,
                "min_importance": min_importance,
                "max_features": int(max_features),
                "gov_policy_events": neutralize_prompt_braces(gov_str),
                "monetary_policy_events":
                neutralize_prompt_braces(monetary_str),
                "cctv_news_events": neutralize_prompt_braces(cctv_str),
                "news_events": neutralize_prompt_braces(event_str),
                "json_structure": get_fewshot_template2(TextualFeatureResult)
            }
            thought = thoughts1[thoughts_name]
            thougths_cot = thought["cot"]

            prompt = f"""
            {thougths_cot}
            \n\n输出格式必须为:\n
                {{json_structure}}
            """
            tasks.append(
                generate_with_semaphore(trade_time=dt,
                                        agent_instance=agent,
                                        semaphore=self.semaphore,
                                        human_message=prompt,
                                        params_dict=params,
                                        schema_cls=TextualFeatureResult,
                                        key_name='textuals'))
            if len(tasks) >= 1:
                await run_task(tasks)
                tasks = []
        if len(tasks) > 0:
            await run_task(tasks)
            
        batch_results = await asyncio.gather(*tasks)
        for result in batch_results:
            trade_date = result['output']['trade_time']
            textuals_data = pd.DataFrame(result['output']['textuals'])
            filename = os.path.join(base_dirs,
                                    "{0}.feather".format(trade_date))
            textuals_data.to_feather(filename)
            # textuals_data.to_feather(
            #     "records/data/event/textuals/{0}.feather".format(trade_date))
