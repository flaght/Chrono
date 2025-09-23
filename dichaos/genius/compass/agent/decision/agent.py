import asyncio, pdb
from typing import Dict
from dichaos.kdutils.logger import logger
from dichaos.agents.agents import Agents as BaseAgents
from .model import *
from .prompt import *


class DecisionAgent(BaseAgents):
    name = 'decision'

    def __init__(self, name: str, top_k: int, vector_provider: str,
                 db_name: str, embedding_model: str, embedding_provider: str,
                 llm_model: str, llm_provider: str, memory_params: Dict):
        super(DecisionAgent,
              self).__init__(name=name,
                             top_k=top_k,
                             vector_provider=vector_provider,
                             db_name=db_name,
                             embedding_model=embedding_model,
                             embedding_provider=embedding_provider,
                             llm_model=llm_model,
                             llm_provider=llm_provider,
                             memory_params=memory_params,
                             system_message=system_message)

    # ### 新增：专门用于综合辩论的异步方法
    async def agenerate_prediction(self, debate_transcript: str, symbol: str,
                                   date: str):
        DomInfo3 = create_decision_dom()
        for i in range(5):
            try:
                response = await self.agenerate_message(
                    decision_human_message,
                    params={
                        "ticker": symbol,
                        "date": date,
                        "debate_transcript": debate_transcript
                    },
                    default={},
                    response_schema=DomInfo3)

                # 检查响应是否有效
                if response and hasattr(
                        response,
                        '__dict__') and 'reasoning' in response.__dict__:
                    break  # 成功，跳出循环
                else:
                    logger.info(f'Retrying... (Attempt {i+1}/5)')
                    await asyncio.sleep(5)  # 使用异步休眠，不会阻塞事件循环
            except Exception as e:
                logger.info(f'Error on attempt {i+1}/5: {e}')
                await asyncio.sleep(5)  # 发生错误时也使用异步休眠

        return response
