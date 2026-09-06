from langchain_community.document_loaders import MongodbLoader as BaseMongoLoader
from pymongo import UpdateOne, InsertOne, UpdateMany, ReturnDocument
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from datetime import datetime, date


class MongoLoader(BaseMongoLoader):

    async def afind(self,
                    query,
                    collection_name=None,
                    limit=100,
                    key_or_list=[],
                    projection=None):
        collection_name = collection_name if collection_name else self.collection_name

        # 构建查询，projection 作为 find 的第二个参数
        if projection:
            find_query = self.db[collection_name].find(query, projection)
        else:
            find_query = self.db[collection_name].find(query)

        if len(key_or_list) == 0:
            cursor = find_query.limit(limit)
        else:
            cursor = find_query.sort(key_or_list).limit(limit)
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


class MongoFactory(object):

    def __init__(self, connection_string: str, db_name: str,
                 collection_name: str):
        self.loader = MongoLoader(connection_string=connection_string,
                                  db_name=db_name,
                                  collection_name=collection_name)
        self.db_name = db_name
        self.collection_name = collection_name

    #### 同步操作 ###
    def insert_one(self,
                   doc: Dict[str, Any],
                   collection_name: Optional[str] = None):
        """
        Insert a single document using bulk_write for atomicity.
        """
        if not doc:
            raise ValueError("doc 不能为空")

        collection_name = collection_name if collection_name else self.collection_name
        request = InsertOne(doc)
        return self.loader.bulk([request], collection_name=collection_name)

    def update_many(self,
                    filter_query: Dict[str, Any],
                    update_doc: Dict[str, Any],
                    collection_name: Optional[str] = None,
                    upsert: bool = False):
        """
        Update many documents. Mirrors pymongo.UpdateMany usage in tests.
        """
        if not filter_query or not update_doc:
            raise ValueError("filter_query 和 update_doc 不能为空")

        collection_name = collection_name if collection_name else self.collection_name
        request = UpdateMany(filter_query, update_doc, upsert=upsert)
        return self.loader.bulk([request], collection_name=collection_name)

    def upsert_topic(self,
                     doc: Dict[str, Any],
                     collection_name: Optional[str] = None):
        """
        Update existing doc (matched by _id or id) or insert if missing.
        """
        if not doc:
            raise ValueError("doc 不能为空")

        collection_name = collection_name if collection_name else self.collection_name

        cluster_id = doc.get("cluster_id")
        doc_id = doc.get("id")
        if not cluster_id:
            raise ValueError("doc 必须包含 cluster_id 作为唯一索引的一部分")
        if not doc_id:
            raise ValueError("doc 必须包含 id 作为唯一索引的一部分")

        update_doc = doc.copy()
        insert_doc = {}

        if "_id" in update_doc:
            insert_doc["_id"] = update_doc.pop("_id")

        update_payload = {"$set": update_doc}
        if insert_doc:
            update_payload["$setOnInsert"] = insert_doc

        request = UpdateOne({
            "cluster_id": cluster_id,
            "id": doc_id
        },
                            update_payload,
                            upsert=True)
        return self.loader.bulk([request], collection_name=collection_name)

    def update_insight(self,
                       cluster_id,
                       new_topic,
                       summary=None,
                       old_topic=None,
                       change_reason=None):
        """
        Update primary_topic_snapshot in a separate collection 'apex_insight'.
        Optionally update summary and record topic change history in 'apex_topic_changes'.
        
        Args:
            cluster_id: 集群ID
            new_topic: 新的主题名称
            summary: 题材摘要（可选）
            old_topic: 旧的主题名称（可选，用于记录变更历史）
            change_reason: 变更原因（可选，用于记录变更历史）
        """
        # [FIX] Write to separate collection
        update_payload = {
            "primary_topic": new_topic,
            "updated_at": datetime.now()
        }
        if summary:
            update_payload["summary"] = summary

        # 记录主题变更历史到独立的集合 'apex_topic_changes'
        if old_topic and old_topic != new_topic:
            try:
                change_record = {
                    "cluster_id": cluster_id,
                    "old_topic": old_topic,
                    "new_topic": new_topic,
                    "change_reason": change_reason or "重聚合更新",
                    "changed_at": datetime.now()
                }

                # 插入到独立的变更记录集合
                change_request = InsertOne(change_record)
                self.loader.bulk([change_request], "apex_topic_changes")
                print(
                    f"  [MongoDB] Recorded topic change for {cluster_id}: '{old_topic}' -> '{new_topic}' in 'apex_topic_changes'"
                )
            except Exception as e:
                print(f"  [MongoDB] Error recording topic change: {e}")

        requests = [
            UpdateMany({"cluster_id": cluster_id}, {"$set": update_payload},
                       upsert=True)
        ]
        try:
            self.loader.bulk(requests, "apex_insight")
            summary_msg = f" and summary" if summary else ""
            change_msg = f", change recorded" if old_topic and old_topic != new_topic else ""
            print(
                f"  [MongoDB] Updated Primary Topic{summary_msg}{change_msg} for {cluster_id} to '{new_topic}' in 'apex_insight'"
            )
        except Exception as e:
            print(f"  [MongoDB] Update Error: {e}")

    def fetch_topic_changes(self,
                            cluster_id: str,
                            limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取指定集群的主题变更历史记录。
        
        Args:
            cluster_id: 集群ID
            limit: 返回的最大记录数，默认50条
            
        Returns:
            主题变更记录列表，按时间倒序排列
        """
        try:
            col = self.loader.client.delegate[
                self.loader.db_name]["apex_topic_changes"]

            cursor = col.find({
                "cluster_id": cluster_id
            }).sort("changed_at", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"  [MongoDB] Get Topic Changes Error: {e}")
            return []

    def get_cluster_events(self,
                           cluster_id: str,
                           start_date: Optional[date] = None,
                           end_date: Optional[date] = None,
                           collection_name: Optional[str] = None,
                           limit: int = 2000) -> pd.DataFrame:
        """
        Return a DataFrame of events for a cluster with optional date filtering.
        """
        if not cluster_id:
            return pd.DataFrame()
        collection_name = collection_name if collection_name else self.collection_name
        query: Dict[str, Any] = {"cluster_id": cluster_id}

        def _normalize(dt_value: Any, is_start: bool) -> Optional[datetime]:
            if dt_value is None:
                return None
            if isinstance(dt_value, datetime):
                return dt_value
            if isinstance(dt_value, date):
                boundary = datetime.min.time(
                ) if is_start else datetime.max.time()
                return datetime.combine(dt_value, boundary)
            # fallback for str / pandas Timestamp
            parsed = pd.to_datetime(dt_value, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()

        start_dt = _normalize(start_date, True)
        end_dt = _normalize(end_date, False)

        date_query: Dict[str, Any] = {}
        if start_dt:
            date_query["$gte"] = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        if end_dt:
            date_query["$lte"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        if date_query:
            query["timestamp"] = date_query

        return self.loader.find(query, collection_name, limit=limit)

    def get_cluster_aliases(self,
                            cluster_id: str,
                            collection_name: Optional[str] = None,
                            limit: int = 1000) -> List[str]:
        """
        Fetch distinct topics for a given cluster_id.
        """
        if not cluster_id:
            return []

        collection_name = collection_name if collection_name else self.collection_name
        df = self.loader.find({"cluster_id": cluster_id},
                              collection_name=collection_name,
                              limit=limit)
        if df.empty or "topic" not in df.columns:
            return []
        return df["topic"].dropna().unique().tolist()

    def find_cluster_id_by_text(
            self,
            embedding_text: str,
            collection_name: Optional[str] = None) -> Optional[str]:
        """
        Find cluster_id by matching embedding_text.
        """
        if not embedding_text:
            return None

        collection_name = collection_name if collection_name else self.collection_name
        df = self.loader.find({"embedding_text": embedding_text},
                              collection_name=collection_name,
                              limit=1)
        if df.empty or "cluster_id" not in df.columns:
            return None
        return df.iloc[0]["cluster_id"]

    def fetch_news_by_cluster(self,
                              cluster_id: str,
                              collection_name: Optional[str] = None,
                              limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Return ordered news documents for the specified cluster.
        """
        if not cluster_id:
            return []

        collection_name = collection_name if collection_name else self.collection_name
        df = self.loader.find({"cluster_id": cluster_id},
                              collection_name=collection_name,
                              limit=limit)
        if df.empty:
            return []

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        return df.to_dict("records")

    def update_cluster_states(
            self,
            cluster_id: str,
            timestamp: datetime,
            stocks: List[str],
            delta_score: float,
            is_new: bool = False,
            primary_topic: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # 1. Prepare Update Operations
        update_ops = {
            "$inc": {
                "count": 1
            },
            "$max": {
                "end_time": timestamp
            },
            "$min": {
                "start_time": timestamp
            },  # Only updates if timestamp < existing start_time
            "$addToSet": {
                "related_stocks": {
                    "$each": stocks if stocks else []
                }
            },
            "$set": {
                "updated_at": datetime.now()
            }
        }

        # If new cluster, set initial fields
        if is_new and primary_topic:
            update_ops["$setOnInsert"] = {
                "primary_topic": primary_topic,
                "created_at": timestamp
            }

        # 2. Execute Atomic Update
        # We use find_one_and_update to get the updated document (for re-aggregation check)
        try:
            # Access the collection via synchronous client (like find() and bulk() methods)
            col = self.loader.client.delegate[
                self.loader.db_name]["apex_states"]

            updated_doc = col.find_one_and_update(
                {"_id": cluster_id},
                update_ops,
                upsert=True,
                return_document=ReturnDocument.AFTER)

            # 3. Update Stock Weights (Nested fields)
            # This is tricky with atomic operators for dynamic keys.
            # We can use bulk write for stock weights to avoid pulling the whole dict.
            if stocks and delta_score > 0:
                weight_ops = []
                for code in stocks:
                    # inc score, inc count, set last_update
                    weight_ops.append(
                        UpdateOne({"_id": cluster_id}, {
                            "$inc": {
                                f"stock_weights.{code}.score": delta_score,
                                f"stock_weights.{code}.count": 1
                            },
                            "$set": {
                                f"stock_weights.{code}.last_update": timestamp
                            }
                        }))
                if weight_ops:
                    col.bulk_write(weight_ops, ordered=False)

            return updated_doc

        except Exception as e:
            print(f"  [MongoDB] Atomic Update Error: {e}")
            return None

    def refresh_cluster_states(self, cluster_id: str, new_topic: str) -> bool:
        """
        Update the primary_topic field in apex_states collection.
        Returns True if successful, False otherwise.
        """
        if not cluster_id or not new_topic:
            return False

        try:
            # Access the collection via synchronous client (consistent with other methods)
            col = self.loader.client.delegate[
                self.loader.db_name]["apex_states"]

            result = col.update_one({"_id": cluster_id}, {
                "$set": {
                    "primary_topic": new_topic,
                    "updated_at": datetime.now()
                }
            })
            return result.modified_count > 0 or result.matched_count > 0
        except Exception as e:
            print(f"  [MongoDB] Update Primary Topic Error: {e}")
            return False

    def fetch_cluster_state(self, cluster_id):
        """
        Fetch single cluster state from 'apex_states'.
        """
        try:
            # Access the collection via synchronous client (consistent with other methods)
            col = self.loader.client.delegate[
                self.loader.db_name]["apex_states"]
            return col.find_one({"_id": cluster_id})
        except Exception as e:
            print(f"  [MongoDB] Get State Error: {e}")
            return None

    #### 异步查询 ####

    async def afind(self,
                    query: Dict[str, Any],
                    collection_name: Optional[str] = None,
                    limit: int = 100,
                    key_or_list: Optional[List] = None,
                    projection: Optional[Dict[str, Any]] = None):
        """
        异步查询指定集合，返回 DataFrame，与 MongoLoader.afind 保持一致。
        """
        collection_name = collection_name if collection_name else self.collection_name
        key_or_list = key_or_list or []
        return await self.loader.afind(query=query,
                                       collection_name=collection_name,
                                       limit=limit,
                                       key_or_list=key_or_list,
                                       projection=projection)

    async def _fetch(self,
                     query: dict,
                     projection: dict,
                     sort_key: list,
                     collection_name: Optional[str] = None,
                     limit: int = 20):
        collection_name = collection_name if collection_name else self.collection_name
        result = await self.loader.afind(query=query,
                                         collection_name=collection_name,
                                         limit=limit,
                                         key_or_list=sort_key,
                                         projection=projection)

        return result

    async def afetch_topic_changes(self,
                                   cluster_id: str,
                                   limit: int = 50) -> List[Dict[str, Any]]:
        """
        异步获取指定集群的主题变更历史记录。
        
        Args:
            cluster_id: 集群ID
            limit: 返回的最大记录数，默认50条
            
        Returns:
            主题变更记录列表，按时间倒序排列
        """
        try:
            cursor = self.loader.db["apex_topic_changes"].find({
                "cluster_id":
                cluster_id
            }).sort("changed_at", -1).limit(limit)
            results = []
            async for document in cursor:
                results.append(document)
            return results
        except Exception as e:
            print(f"  [MongoDB] Get Topic Changes Error: {e}")
            return []

    async def fetch_feeds(self,
                          query: dict,
                          projection: dict,
                          sort_key: list,
                          collection_name: Optional[str] = None,
                          limit: int = 20) -> pd.DataFrame:
        collection_name = collection_name if collection_name else self.collection_name

        # 查询条件：processed 为 0
        #query = {"processed": 0}

        # 投影：只返回指定字段
        #projection = {
        #    "_id": 0,  # 排除 _id
        #    "id": 1,
        #    "publish_time": 1,
        #    "title": 1,
        #    "summary": 1,
        #    "url": 1
        #}

        # 排序：根据 create_time 正序排序
        #sort_key = [("create_time", 1)]

        # 执行异步查询
        result = await self.loader.afind(query=query,
                                         collection_name=collection_name,
                                         limit=limit,
                                         key_or_list=sort_key,
                                         projection=projection)

        return result

    async def fetch_events(self,
                           query: dict,
                           projection: dict,
                           sort_key: list,
                           collection_name: Optional[str] = None,
                           limit: int = 20) -> pd.DataFrame:
        result = await self._fetch(query=query,
                                   projection=projection,
                                   sort_key=sort_key,
                                   collection_name=collection_name,
                                   limit=limit)

        return result

    async def refresh(self, data, collection_name=None, batch_size=50000):
        if data is None or data.empty:
            print("DataFrame 为空，没有数据需要写入")
            return None
        collection_name = collection_name if collection_name else self.collection_name
        documents = data.where(pd.notna(data), None).to_dict('records')
        valid_docs = []
        id_list = []
        for doc in documents:
            # 跳过完全为空或无效的文档
            if not doc:
                continue

            # 使用 id 作为唯一标识符（id 肯定存在）
            doc_id = doc.get("id")
            if not doc_id:
                continue  # 跳过没有 id 的文档

            valid_docs.append(doc)
            id_list.append(doc_id)

        if not valid_docs:
            print("没有有效数据需要写入")
            return None

        # 使用 UpdateOne with upsert=True 来避免重复键错误
        # 这样可以安全地处理并发情况，如果文档已存在则更新，不存在则插入
        # 不需要预先查询，因为 upsert 是原子操作，可以避免竞态条件
        requests = []
        for doc in valid_docs:
            doc_id = doc.get("id")
            if not doc_id:
                continue

            # processed 只在首次插入时写入，避免把已处理的记录重置为 0
            update_payload = {"$set": doc}
            processed_value = update_payload["$set"].pop("processed", None)
            if processed_value is not None:
                update_payload["$setOnInsert"] = {"processed": processed_value}

            # 使用 upsert 操作：如果文档不存在则插入，存在则更新
            # 这样可以避免并发情况下的重复键错误
            update_op = UpdateOne(
                {"id": doc_id},  # 查询条件
                update_payload,  # 更新内容
                upsert=True  # 如果不存在则插入
            )
            requests.append(update_op)

        if not requests:
            print("没有有效数据需要写入")
            return None

        # 执行异步批量写入（upsert）
        try:
            result = await self.loader.abulk(requests,
                                             collection_name=collection_name)
            print(
                f"批量写入完成: 插入 {result.upserted_count} 条新数据，更新 {result.modified_count} 条已存在数据，匹配 {result.matched_count} 条"
            )
            return result
        except Exception as e:
            print(f"批量写入失败: {e}")
            raise

    async def mark_processed(self,
                             ids: List[str],
                             collection_name: Optional[str] = None):
        if not ids:
            print("没有可更新的 id")
            return None

        collection_name = collection_name if collection_name else self.collection_name

        requests = [
            UpdateOne({"id": doc_id}, {"$set": {
                "processed": 1
            }}) for doc_id in ids if doc_id
        ]

        if not requests:
            print("没有有效的更新请求")
            return None

        try:
            result = await self.loader.abulk(requests,
                                             collection_name=collection_name)
            print(
                f"批量更新 processed 完成: 匹配 {result.matched_count} 条，更新 {result.modified_count} 条"
            )
            return result
        except Exception as e:
            print(f"批量更新 processed 失败: {e}")
            raise

    async def afetch_events_by_time_range(
            self,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            collection_name: Optional[str] = None,
            limit: int = 10000) -> pd.DataFrame:
        """
        批量查询指定时间范围内的所有事件（用于热门主题计算）
        
        Args:
            start_date: 开始时间
            end_date: 结束时间
            collection_name: 集合名称
            limit: 最大返回数量
            
        Returns:
            包含所有事件的 DataFrame
        """
        collection_name = collection_name if collection_name else self.collection_name
        query: Dict[str, Any] = {}

        def _normalize(dt_value: Any, is_start: bool) -> Optional[datetime]:
            if dt_value is None:
                return None
            if isinstance(dt_value, datetime):
                return dt_value
            if isinstance(dt_value, date):
                boundary = datetime.min.time(
                ) if is_start else datetime.max.time()
                return datetime.combine(dt_value, boundary)
            parsed = pd.to_datetime(dt_value, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()

        start_dt = _normalize(start_date, True)
        end_dt = _normalize(end_date, False)

        date_query: Dict[str, Any] = {}
        if start_dt:
            date_query["$gte"] = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        if end_dt:
            date_query["$lte"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        if date_query:
            query["timestamp"] = date_query

        return await self.loader.afind(query, collection_name, limit=limit)

    async def afetch_cluster_events(self,
                                    cluster_id: str,
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None,
                                    collection_name: Optional[str] = None,
                                    limit: int = 2000) -> pd.DataFrame:
        """
        Async version: Return a DataFrame of events for a cluster with optional datetime filtering.
        Supports precision down to seconds (e.g., 2025-01-15T09:30:00).
        """
        if not cluster_id:
            return pd.DataFrame()
        collection_name = collection_name if collection_name else self.collection_name
        query: Dict[str, Any] = {"cluster_id": cluster_id}

        def _normalize(dt_value: Any, is_start: bool) -> Optional[datetime]:
            if dt_value is None:
                return None
            if isinstance(dt_value, datetime):
                return dt_value
            if isinstance(dt_value, date):
                boundary = datetime.min.time(
                ) if is_start else datetime.max.time()
                return datetime.combine(dt_value, boundary)
            # fallback for str / pandas Timestamp
            parsed = pd.to_datetime(dt_value, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()

        start_dt = _normalize(start_date, True)
        end_dt = _normalize(end_date, False)

        date_query: Dict[str, Any] = {}
        if start_dt:
            date_query["$gte"] = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        if end_dt:
            date_query["$lte"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        if date_query:
            query["timestamp"] = date_query

        return await self.loader.afind(query, collection_name, limit=limit)

    async def afetch_news_by_cluster(
            self,
            cluster_id: str,
            collection_name: Optional[str] = None,
            limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Async version: Return ordered news documents for the specified cluster.
        """
        if not cluster_id:
            return []

        collection_name = collection_name if collection_name else self.collection_name
        df = await self.loader.afind({"cluster_id": cluster_id},
                                     collection_name=collection_name,
                                     limit=limit)
        if df.empty:
            return []

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        return df.to_dict("records")

    async def afind_cluster_id_by_text(
            self,
            embedding_text: str,
            collection_name: Optional[str] = None) -> Optional[str]:
        """
        Async version: Find cluster_id by matching embedding_text.
        """
        if not embedding_text:
            return None

        collection_name = collection_name if collection_name else self.collection_name
        df = await self.loader.afind({"embedding_text": embedding_text},
                                     collection_name=collection_name,
                                     limit=1)
        if df.empty or "cluster_id" not in df.columns:
            return None
        return df.iloc[0]["cluster_id"]

    async def afetch_cluster_state(
            self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """
        Async version: Fetch single cluster state from 'apex_states'.
        """
        try:
            # Access the collection via async client
            doc = await self.loader.db["apex_states"].find_one(
                {"_id": cluster_id})
            return doc
        except Exception as e:
            print(f"  [MongoDB] Get State Error: {e}")
            return None

    async def afetch_cluster_states(self,
                                    query: dict = None,
                                    limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch cluster states from 'apex_states' collection using async client.
        Sorted by end_time descending, limited to specified number of results.
        """
        if query is None:
            query = {}

        try:
            # Access the collection via async client
            cursor = self.loader.db["apex_states"].find(query).sort(
                "end_time", -1).limit(limit)
            results = []
            async for document in cursor:
                results.append(document)
            return results
        except Exception as e:
            print(f"  [MongoDB] Fetch Cluster States Error: {e}")
            return []

    async def afind_hot_topics_cursor(self,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None,
                                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Async version: Query hot topics from 'apex_states' with sorting and filtering.
        Returns list of cluster state documents sorted by count descending.
        Overlap logic: document.start_time <= query_end_date AND document.end_time >= query_start_date
        """
        query = {}

        def _normalize(dt_value: Any, is_start: bool) -> Optional[datetime]:
            if dt_value is None:
                return None
            if isinstance(dt_value, datetime):
                return dt_value
            if isinstance(dt_value, date):
                boundary = datetime.min.time(
                ) if is_start else datetime.max.time()
                return datetime.combine(dt_value, boundary)
            parsed = pd.to_datetime(dt_value, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()

        start_dt = _normalize(start_date, True)
        end_dt = _normalize(end_date, False)
        if start_dt and end_dt:
            # Overlap logic: start_time <= end_date AND end_time >= start_date
            # This finds clusters whose time range overlaps with the query range
            # Convert to string format to match database storage format
            query_start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            query_end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            # Document overlaps query range if: doc.start_time <= query_end AND doc.end_time >= query_start
            # Note: This is a preliminary filter. The actual filtering should be done by checking events in the time range.
            query["start_time"] = {"$lte": query_end_str}
            query["end_time"] = {"$gte": query_start_str}

        try:
            # Access the collection via async client
            cursor = self.loader.db["apex_states"].find(query).sort(
                "count", -1).limit(limit)
            results = []
            async for document in cursor:
                results.append(document)
            return results
        except Exception as e:
            print(f"  [MongoDB] Hot Topics Error: {e}")
            return []


    async def afetch_parts(self, names, collection_name):
        query = {"name": {"$in": names}}
        cursor = self.loader.db[collection_name].find(query)
        docs_dict = {}
        async for document in cursor:
            docs_dict[document["name"]] = document

        return expandvars(docs_dict)

    async def afetch_graphs(self, user_id, graph_id, collection_name):
        query = {"user_id": user_id, "graph_id": graph_id}
        doc = await self.loader.db[collection_name].find_one(query)
        return doc