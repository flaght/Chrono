from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
import sqlalchemy.orm as orm
import math
import numpy as np
import pandas as pd


class DataAdapter(object):

    @classmethod
    def create_adapter(cls, uri):
        category, _ = uri.split('://')
        if 'sql' in category:
            return DatabaseAdapter(uri=uri)

    def __init__(self, uri):
        self._uri = uri

    def clear_data(self, **kwargs):
        raise NotImplementedError

    def refresh_data(self, **kwargs):
        raise NotImplementedError

    def all_clear(self, **kwargs):
        raise NotImplementedError


class DatabaseAdapter(DataAdapter):
    __name__ = 'database'

    def __init__(self, uri):
        super(DatabaseAdapter, self).__init__(uri=uri)
        self._db_client = create_engine(uri, echo=False)
        self._db_session = self.create_session()
        self._base = automap_base()
        self._base.prepare(self._db_client, reflect=True)

    def create_session(self):
        db_session = orm.sessionmaker(bind=self._db_client)
        return db_session()

    def __enter__(self):
        return self

    def clear_data(self, **kwargs):
        if 'value' not in kwargs:
            return
        key = 'id' if 'key' not in kwargs else kwargs['key']
        values = [str(v) for v in kwargs['value']]
        ids = "\'" + "\',\'".join(values) + "\'"
        sql = f"""delete from {kwargs['table_name']}  where {key} in ({ids}) and flag = 1;"""
        self._db_session.execute(sql)
        self._db_session.commit()
        self._db_session.close()

    def _refresh_data(self, **kwargs):
        table_name = kwargs['table_name']
        sql = "INSERT INTO {0} SET".format(kwargs['table_name'])
        columns = kwargs['columns'] if 'columns' in kwargs else None
        if columns is None:
            updates = ",".join("{0} = :{0}".format(x)
                               for x in list(kwargs['df']))
            sql = sql + "\n" + updates
            sql = sql + "\n" + "ON DUPLICATE KEY UPDATE"
            sql = sql + "\n" + updates
        else:
            updates = ",".join("{0} = :{0}".format(x) for x in columns)
            sql = sql + "\n" + updates
            sql = sql + "\n" + "ON DUPLICATE KEY UPDATE"
            sql = sql + "\n" + updates
        for index, row in kwargs['df'].iterrows():
            dictInput = dict(row)
            self._db_session.execute(sql, dictInput)
        self._db_session.commit()
        self._db_session.close()

    def _update_data(self, **kwargs):
        print("""refresh mysql {0}""".format(kwargs['table_name']))
        total_data = kwargs['total_data']
        count = 2000 if 'count' not in kwargs else int(kwargs['count'])
        #if 'trade_date' in total_data.columns:
        #    total_data['trade_date'] = pd.to_datetime(
        #        total_data['trade_date']).dt.strftime('%Y-%m-%d')
        total_data = total_data.replace([np.inf, -np.inf, "NaT"], np.nan)
        total_data = total_data.where(pd.notnull(total_data), None)
        total_data = total_data.replace({np.nan: None})

        #total_data = total_data.replace({pd.NA: None})
        total_count = len(total_data)
        page = math.ceil(total_count / count)
        for pos in range(0, page):
            print(
                "update table:{0},pos:{1},page:{2},total:{3},count:{4}".format(
                    kwargs['table_name'], pos, page, total_count, count))
            self._refresh_data(df=total_data[pos * count:(pos + 1) * count],
                               table_name=kwargs['table_name'])

    def _import_data(self, **kwargs):
        if 'flag' in kwargs['total_data'].columns:
            total_data = kwargs['total_data'].dropna(subset=['flag'])
            total_data = total_data[total_data.flag == 1]
        else:
            total_data = kwargs['total_data']
        total_data = total_data.replace([np.inf, -np.inf, "NaT"], np.nan)
        total_data = total_data.where(pd.notnull(total_data), None)
        if not total_data.empty:
            count = 200 if 'count' not in kwargs else kwargs['count']
            page = math.ceil(len(total_data) / count)
            for pos in range(0, page):
                total_data[pos * count:(pos + 1) * count].to_sql(
                    name=kwargs['table_name'],
                    con=self._db_client,
                    if_exists='append',
                    index=False)

    def name(self, name):
        return None if name not in self._base.classes else self._base.classes[
            name]

    def refresh_data(self, **kwargs):  #increment #full
        if 'trade_date' in kwargs['total_data'].columns:
            kwargs['total_data']['trade_date'] = pd.to_datetime(
                kwargs['total_data']['trade_date']).dt.strftime('%Y-%m-%d')
        if kwargs['method'] == 'full':
            self._import_data(**kwargs)
        elif kwargs['method'] == 'increment':
            self._update_data(**kwargs)


data_adapter = DataAdapter.create_adapter(
    'mysql+mysqlconnector://neutron:Jc2D6sip@172.17.0.1:3306/quant')
