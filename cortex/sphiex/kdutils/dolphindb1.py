import re, traceback, math
import pandas as pd
import numpy as np
import dolphindb


class DolphinDBAdapter(object):
    __name__ = 'ddb'

    def _parser(self, url):
        #pattern = r"ddb://([^:]+):(.+?)@([^:]+):(\d+)"
        pattern = r"ddb://([^:]+):(.*)@([^@:]+):(\d+)"
        match = re.match(pattern, url)
        user = match.group(1)
        password = match.group(2)
        address = match.group(3)
        port = match.group(4)
        return user, password, address, int(port)

    def get_table_schema(self, db_path, table_name):
        sql = """schema(loadTable('{db_name}',`{table}))""".format(
            db_name=db_path, table=table_name)
        data = self._ddb_client.run(sql)
        col_list = data['colDefs']['name'].to_list()
        return col_list

    def allign_data_types(self, dbPath, tableName, df):
        sql = """schema(loadTable('{dbName}',`{table}))""".format(
            dbName=dbPath, table=tableName)
        data = self._ddb_client.run(sql)
        type_dict = dict(
            zip(data['colDefs']['name'], data['colDefs']['typeString']))

        def as_type(tmp, col, type):
            target_type = type_dict[col]
            if target_type == 'DATE' or target_type == 'TIMESTAMP' or target_type == 'DATETIME':
                if type == object:  # str
                    tmp = pd.to_datetime(tmp)
                elif type == 'int64':  #20230101
                    tmp = pd.to_datetime(tmp.astype(str))
            elif target_type == 'DOUBLE':
                tmp = tmp.astype(np.float64)
            elif target_type == 'LONG':
                tmp = tmp.astype('Int64')
            elif target_type == 'INT':
                try:
                    tmp = tmp.astype('Int32')
                except:
                    tmp = tmp.astype('float').astype('Int32')
            elif target_type == 'SYMBOL' or target_type == 'STRING':
                tmp.fillna('', inplace=True)
                tmp = tmp.astype(str)
            return tmp

        for col, type in df.dtypes.items():
            if col not in type_dict.keys():
                continue
            df[col] = as_type(df[col], col, type)

        col_list = data['colDefs']['name'].to_list()
        try:
            df = df[col_list]
        except:
            print(traceback.format_exc(), tableName)

        return df

    def __init__(self, uri):
        self._ddb_client = dolphindb.session()
        user, password, address, port = self._parser(uri)
        self._ddb_client.connect(host=address,
                                 port=port,
                                 userid=user,
                                 password=password)

    def _update_data(self, **kwargs):
        table_name = kwargs['table_name']
        total_data = kwargs['total_data']
        if 'db_name' in kwargs and isinstance(kwargs['db_name'], str):
            dbPath = "dfs://{0}".format(kwargs['db_name'])
        else:
            dbPath = 'dfs://' + self.to_ddb_table(table_name)
        # dbPath = 'dfs://' + self.to_ddb_table(table_name)
        if self._ddb_client.existsDatabase(dbPath):
            if self._ddb_client.existsTable(dbPath, table_name):
                try:
                    if len(total_data) > 0:
                        df = self.allign_data_types(dbPath, table_name,
                                                    total_data)
                        appender = dolphindb.tableAppender(
                            dbPath=dbPath,
                            tableName=table_name,
                            ddbSession=self._ddb_client)
                        num1 = df.shape[0]
                        base1 = 2000000
                        count1 = math.ceil(num1 / base1)
                        df = df.reset_index(drop=True)
                        for count in range(0, count1):
                            num = appender.append(
                                df.loc[count * base1:(count + 1) * base1])
                except:
                    print("insert df, total size in bytes: %d" %
                          total_data.memory_usage().sum())
                    print(traceback.format_exc(), table_name)

    def refresh_data(self, **kwargs):
        if kwargs['method'] == 'full':
            self._update_data(**kwargs)
        elif kwargs['method'] == 'increment':
            self._update_data(**kwargs)
