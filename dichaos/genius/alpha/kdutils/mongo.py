from langchain_community.document_loaders import MongodbLoader as BaseMongoLoader
import pandas as pd
import pdb, asyncio


class MongoLoader(BaseMongoLoader):

    async def afind(self,
                    query,
                    collection_name=None,
                    limit=100,
                    key_or_list=[]):
        collection_name = collection_name if collection_name else self.collection_name
        if len(key_or_list) == 0:
            cursor = self.db[collection_name].find(query).limit(limit)
        else:
            cursor = self.db[collection_name].find(query).sort(
                key_or_list).limit(limit)
        count = []
        async for document in cursor:
            count.append(document)

        dataframes = pd.DataFrame(count)
        return dataframes

    def find(self, query, collection_name=None, limit=100, key_or_list=[]):
        collection_name = collection_name if collection_name else self.collection_name
        if len(key_or_list) == 0:
            cursor = self.client.delegate[self.db_name][collection_name].find(
                query).limit(limit)
        else:
            cursor = self.client.delegate[self.db_name][collection_name].find(
                query).sort(key_or_list).limit(limit)
        dataframes = pd.DataFrame(cursor)
        return dataframes

    def bulk(self,
             requests,
             collection_name=None,
             bypass_document_validation=True):
        collection_name = collection_name if collection_name else self.collection_name
        result = self.client.delegate[
            self.db_name][collection_name].bulk_write(
                requests,
                bypass_document_validation=bypass_document_validation)
        return result

    async def abulk(self, requests, collection_name=None):
        collection_name = collection_name if collection_name else self.collection_name
        result = await self.db[collection_name].bulk_write(requests)
        return result
