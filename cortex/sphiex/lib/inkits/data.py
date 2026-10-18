import pdb
from kdutils.mongo import MongoFactory as BaseFactory
from kdutils.tools import expandvars


class MongoFactory(BaseFactory):

    def __init__(self, connection_string: str, db_name: str,
                 collection_name: str):
        super(MongoFactory, self).__init__(connection_string=connection_string,
                                           db_name=db_name,
                                           collection_name=collection_name)

    ## 查询武器库
    async def _afetch_parts(self, names, collection_name):
        query = {"name": {"$in": names}}
        cursor = self.loader.db[collection_name].find(query)
        docs_dict = {}
        async for document in cursor:
            docs_dict[document["name"]] = document

        return expandvars(docs_dict)



    ## 查询graphs
    async def afetch_graphs(self, user_id, graph_id, collection_name):
        query = {"user_id": user_id, "graph_id": graph_id}
        doc = await self.loader.db[collection_name].find_one(query)
        return doc

    async def afetch_system_llm(self, names):
        return await self._afetch_parts(names=names,
                                        collection_name='inkits_system_llm')

    async def afetch_system_vector(self, names):
        return await self._afetch_parts(names=names,
                                        collection_name='inkits_system_vector')

    async def afetch_system_persona(self, names):
        return await self._afetch_parts(
            names=names, collection_name='inkits_system_persona')


    async def afetch_system_thoughts(self, names):
        return await self._afetch_parts(
            names=names, collection_name='inkits_system_thoughts')


    async def afetch_graphs_docs(self, user_id, graph_id):

        graphs_docs = await self.afetch_graphs(
            user_id=user_id,
            graph_id=graph_id,
            collection_name="inkits_strategy_graphs")
        ## 发散 agent
        llm_list = [agent['llm'] for agent in graphs_docs['diver']['agents']]
        vector_list = [
            agent['vector'] for agent in graphs_docs['diver']['agents']
        ]
        persona_list = [
            agent['persona'] for agent in graphs_docs['diver']['agents']
        ]

        ## 博弈 agent
        llm_list += [agent['llm'] for agent in graphs_docs['battle']['agents']]
        vector_list += [
            agent['vector'] for agent in graphs_docs['battle']['agents']
        ]
        persona_list += [
            agent['persona'] for agent in graphs_docs['battle']['agents']
        ]

        ## 路由agent
        llm_list += [graphs_docs['directent']['agents']['llm']]
        vector_list += [graphs_docs['directent']['agents']['vector']]
        persona_list += [graphs_docs['directent']['agents']['persona']]

        llm_list = list(set(llm_list))
        vector_list = list(set(vector_list))
        persona_list = list(set(persona_list))

        llm_profiles = await self.afetch_system_llm(names=llm_list)
        vector_profiles = await self.afetch_system_vector(names=vector_list)
        persona_profiles = await self.afetch_system_persona(names=persona_list)
        
        return graphs_docs, llm_profiles, vector_profiles, persona_profiles

    async def afetch_strategy_graphs(self, user_id, collection_name='inkits_strategy_graphs'):
        """查询某个用户下所有的策略图谱列表"""
        query = {"user_id": user_id}
        cursor = self.loader.db[collection_name].find(query)
        docs = []
        async for doc in cursor:
            doc['_id'] = str(doc['_id'])  # ObjectId -> str，避免序列化报错
            docs.append(doc)
        return docs

    async def afetch_user(self, username: str, collection_name='inkits_users'):
        """按用户名查询用户文档（含 uid、password_hash、role 等）"""
        doc = await self.loader.db[collection_name].find_one({"username": username})
        if doc:
            doc['_id'] = str(doc['_id'])
        return doc