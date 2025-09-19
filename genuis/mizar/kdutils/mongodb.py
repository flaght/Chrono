# -*- coding: utf-8 -*-
import logging,gevent,pymongo,time,pdb
import gevent.monkey
from bson.objectid import ObjectId

log = logging.getLogger(__name__)


class MongoDBManager(pymongo.MongoClient):
    """
    MongodbDB 管理类
    """

    def __init__(self, uri=None, conf=None):
        if uri is not None:
            super(MongoDBManager, self).__init__(uri)
        elif conf is not None:
            self._conf = conf
            if '://' in self._conf.get('host'):
                super(MongoDBManager, self).__init__(
                    host=self._conf.get('host')
                )
            else:
                super(MongoDBManager, self).__init__(
                    host=self._conf.get('host'),
                    port=int(self._conf.get('port'))
                )
            if 'user' in self._conf and 'password' in self._conf:
                self[self._conf.get('db')].authenticate(self._conf.get('user'), self._conf.get('password'))
        self.status = type('MongoDBStatus', (), {'alive_timestamp': 0})
        if 'sys' in gevent.monkey.saved:
            gevent.spawn(self._alive_check)
            gevent.sleep()

    def _alive_check(self):
        """
        MongoDB 存活检测
        """
        while True:
            if not self['local']:
                gevent.sleep(5)
                continue
            self.status.alive_timestamp = int(time.time())
            gevent.sleep(5)

    def get_connection_status(self):
        return self.status

    def insert(self, db, table, docs):
        return self[db][table].insert(docs)
    
    def insert_many(self, db, table, docs):
        return self[db][table].insert_many(docs)

    def delete(self, db, table, query):
        return self[db][table].delete_many(query)

    def delete_with_id(self, db, table, oid):
        return self[db][table].delete_many({'_id': ObjectId(oid)})

    def update_many(self, db, table, docs, query=None):
        return self[db][table].update_many(filter=query, update=docs, upsert=True)

    def find_one(self, db, table, query=None):
        return self[db][table].find_one(query)

    def find_one_with_id(self, db, table, oid):
        return self[db][table].find_one({'_id': ObjectId(oid)})

    def find_all(self, db, table, query=None):
        return self[db][table].find(query)