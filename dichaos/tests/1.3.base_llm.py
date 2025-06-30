import sys, os, pdb
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath('..'))
load_dotenv()

from dichaos.services.llm import LLMServiceFactory

memory_id_extract_prompt = "Provide the piece of information related the most to the investment decisions from mainstream sources such as the investment suggestions major fund firms such as ARK, Two Sigma, Bridgewater Associates, etc. in the {memory_layer} memory?"
short_memory_id_desc = "The id of the short-term information."
long_memory_id_desc = "The id of the long-term information."
investment_info_prefix = "The current date is {cur_date}. Here are the observed financial market facts: for {symbol}, the price difference between the next trading day and the current trading day is: {future_record}\n\n"

cur_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this? Just summarize the reason of the decision。
    Your should provide a summary information and the id of the information to support your summary.

    ${investment_info}

    Your output should strictly conforms the following json format without any additional contents: 
    {{
        "summary_reason": "string" 
        "short_memory_index":  [{{"memory_index": 4}}], 
        "middle_memory_index": [{{"memory_index": 4}}], 
        "long_memory_index": [{{"memory_index": 4}}], 
        "reflection_memory_index": [{{"memory_index": 4}}]
    }}
"""


class Memory(BaseModel):
    memory_index: int = Field(
        ...,
        description=memory_id_extract_prompt.format(
            memory_layer='short-level'))


class InvestInfo(BaseModel):
    short_memory_index: List[Memory] = Field(
        ...,
        description=short_memory_id_desc,
    )
    long_memory_index: List[Memory] = Field(
        ...,
        description=short_memory_id_desc,
    )


cur_date = "2022-01-01"
symbol = "TSLA"
future_record = 1.0

short_memory_id = [1]
long_memory_id = [3]

short_memory = ["short memory"]
long_memory = ["long memory"]

llm_service = LLMServiceFactory.create_llm_service(
    base_url=os.getenv('LLM_OPENAI_BASE_URL'),
    api_key=os.getenv('LLM_OPENAI_API_KEY'),
    llm_provider='openai',
    llm_model='deepseek-chat',
    system_message=
    """You are a helpful assistant only capable of communicating with valid JSON, and no other text.
                """)

model = llm_service()
response_model = InvestInfo(short_memory_index=[Memory(memory_index=1)],
                            long_memory_index=[Memory(memory_index=2)])
response = llm_service.structured_output(model=model, response=response_model)

investment_info = investment_info_prefix.format(cur_date=cur_date,
                                                symbol=symbol,
                                                future_record=future_record)

investment_info += "The short-term information:\n"
investment_info += "\n".join(
    [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)])
investment_info += "\n\n"

investment_info += "The long-term information:\n"
investment_info += "\n".join(
    [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)])
investment_info += "\n\n"

prompt = llm_service.create_prompt(cur_prompt,
                                   params={'investment_info': investment_info})

model = llm_service.structured_output(model=model,
                                      response=response_model,
                                      method='json_mode')
pdb.set_trace()
result = model.invoke(prompt)
print(result)