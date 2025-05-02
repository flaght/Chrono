import re, os, pdb
import pandas as pd
import dateutil.parser as dt_parser
from dotenv import load_dotenv

load_dotenv()

import dolphindb as ddb

from alphacopilot.calendars.api import makeSchedule


def convert_date(date, format='%Y.%m.%d'):
    try:
        if isinstance(date, (str)):
            date = dt_parser.parse(date)
    except Exception as e:
        raise Exception('date:{}格式不能识别。' % date)

    return "date('{0}')".format(date.strftime(format))


def to_format(key, express, values):
    if express == 'in':
        values = ', '.join(
            map(lambda x: f"'{x}'" if isinstance(x, str) else f"{x}", values))
        return "{0} {1} [{2}]".format(key, express, values)
    else:
        values = values
        return "{0} {1} ({2})".format(key, express, values)


def parser(url):
    pattern = r"ddb://([^:]+):(.+)@([^:]+):(\d+)"
    match = re.match(pattern, url)
    user = match.group(1)
    password = match.group(2)
    address = match.group(3)
    port = match.group(4)
    return user, password, address, int(port)


def fetch_columns(db_name, table_name):
    if not engine.existsDatabase(db_name) or not engine.existsTable(
            db_name, table_name):
        return pd.DataFrame()
    sql = """schema(loadTable('{dbName}',`{table}))""".format(dbName=db_name,
                                                              table=table_name)
    data = engine.run(sql)
    col_list = data['colDefs']['name'].to_list()
    return col_list


def fetch_data(db_name, table_name, columns=None, clause_list=None):
    if not engine.existsDatabase(db_name) or not engine.existsTable(
            db_name, table_name):
        return pd.DataFrame()
    column_list = fetch_columns(db_name, table_name)
    if columns is not None:
        select_col = ''
        for col in columns:
            if col in column_list:
                select_col = select_col + col + ','
            else:
                print('{0} is not {1} column'.format(col, table_name))
        select_col = select_col[:-1]
    else:
        select_col = '*'

    sql = """select {select_col} from loadTable('{dbName}',`{table}) where 1==1""".format(
        dbName=db_name, table=table_name, select_col=select_col)
    if clause_list is not None:
        for clause in clause_list:
            sql = sql + ' and ' + clause
    data = engine.run(sql)
    return data


begin_date = '2021-01-01'
end_date = '2025-04-10'
dates = makeSchedule(begin_date, end_date, '1b', 'china.sse')

pdb.set_trace()
url = os.environ['DDB_URL']
engine = ddb.session()
user, password, address, port = parser(url)
engine.connect(host=address, port=port, userid=user, password=password)

db_name = "dfs://min_bar"
table_name = "min_snapshot_n_stats"

for table_name in ['min_snapshot_money_flow',
        'min_snapshot_n_stats', 'min_snapshot_order_flow',
        'min_snapshot_price_volume_corr',
        'min_snapshot_price_volume_imbalance', 'min_trade_active_buy_sell',
        'min_trade_different_price_classify', 'min_trade_order_numbers',
        'min_trade_small_big_orders'
][1:]:
    for date in dates:
        print(table_name, date)
        clause_list1 = to_format('date', '>=',
                                 convert_date(date.strftime('%Y-%m-%d')))
        clause_list2 = to_format('date', '<=',
                                 convert_date(date.strftime('%Y-%m-%d')))
        data = fetch_data(db_name=db_name,
                          table_name=table_name,
                          clause_list=[clause_list1, clause_list2])
        if data.empty:
            print('没有数据')
            continue
        path1 = os.path.join("/workspace/data/data/dev/chaos/min_factors", table_name)
        if not os.path.exists(path1):
            os.makedirs(path1)
        filename = os.path.join(path1, date.strftime('%Y-%m-%d') + '.feather')
        data.to_feather(filename)
