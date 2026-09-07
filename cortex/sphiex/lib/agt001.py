import os,pdb
from dichaos.agents.agents import Agents
from lib.inkits.data import MongoFactory
from lib.pam001 import load_memory_params

async def create_agent(llm_name, vector_name, persona_name, 
                       thoughts_name, agent_name, agent_title,
                       category):
    memory_params = load_memory_params()
    data_client = MongoFactory(
        connection_string=os.environ["MONGO_CONNECTION_STRING"],
        db_name=os.environ["MONGO_DB_NAME"],
        collection_name='test1')
    
    
    llm1 = await data_client.afetch_system_llm(names=[llm_name])

    vector1 = await data_client.afetch_system_vector(names=[vector_name])

    persona1 = await data_client.afetch_system_persona(names=[persona_name])

    thoughts1 = await data_client.afetch_system_thoughts(names=[thoughts_name])
    agent = Agents(
        name=agent_name,
        title=agent_title,
        category=category,
        top_k=10,
        db_name="vector1",
        vector_provider=vector1[vector_name]['vector_provider'],
        embedding_provider=vector1[vector_name]['embedding_provider'],
        embedding_model=vector1[vector_name]['embedding_model'],
        llm_provider=llm1[llm_name]['llm_provider'],
        llm_model=llm1[llm_name]['llm_model'],
        mode=llm1[llm_name]['mode'],
        memory_params=memory_params,
        system_message=persona1[persona_name]['system_prompt'],
        other_parameters=llm1[llm_name]['other_parameters'])
    return agent, thoughts1
