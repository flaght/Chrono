import pdb, os, datetime
import numpy as np
import pandas as pd
from pymongo import InsertOne, DeleteOne

from jdw import DBAPI
from alphacopilot.api.data import RetrievalAPI, ddb_tools, DDBAPI
from kdutils.mongodb import MongoDBManager

kd_engine = DBAPI.FetchEngine.create_engine('kd')


def fetch_local_market(base_path, method, instruments, name):
    dirs = os.path.join(base_path, method, instruments, 'basic')
    filename = os.path.join(dirs, f"{name}_data.feather")
    return pd.read_feather(filename)


def fetch_basic(begin_date, end_date, codes):
    name = 'fut_basic'
    names = DBAPI.CustomizeFactory(kd_engine).name(name=name)
    clause_list = [names.contractObject.in_(codes), names.flag == 1]
    basic_info = DBAPI.CustomizeFactory(kd_engine).custom(
        name=name,
        clause_list=clause_list,
        columns=['contractObject', 'contMultNum', 'listDate'])
    basic_info = basic_info.sort_values(by='listDate',
                                        ascending=False).drop_duplicates(
                                            subset='contractObject',
                                            keep='first')
    return basic_info.rename(columns={'contractObject': 'code'})


def fetch_basic1(begin_date, end_date, symbols):
    name = 'fut_basic'
    names = DBAPI.CustomizeFactory(kd_engine).name(name=name)
    clause_list = [names.code.in_(symbols), names.flag == 1]
    basic_info = DBAPI.CustomizeFactory(kd_engine).custom(
        name=name,
        clause_list=clause_list,
        columns=[
            'contractObject', 'code', 'exchangeCD', 'contMultNum',
            'lastTradeDate'
        ])
    # basic_info = basic_info.sort_values(by='listDate',
    #                                     ascending=False).drop_duplicates(
    #                                         subset='contractObject',
    #                                         keep='first')
    basic_info = basic_info[
        (basic_info['lastTradeDate'] >= begin_date.strftime("%Y-%m-%d"))
        & (basic_info['lastTradeDate'] >= end_date.strftime("%Y-%m-%d"))]
    return basic_info.rename(columns={
        'contractObject': 'code',
        'code': 'symbol'
    })


def fetch_basic2(begin_date,
                 end_date,
                 codes=None,
                 columns=[
                     'contractObject', 'code', 'exchangeCD', 'contMultNum',
                     'lastTradeDate'
                 ]):

    name = 'fut_basic'
    names = DBAPI.CustomizeFactory(kd_engine).name(name=name)
    clause_list = [names.flag == 1]
    if isinstance(codes, list):
        clause_list.append(names.contractObject.in_(codes))
    elif isinstance(codes, str):
        clause_list.append(names.contractObject.in_([codes]))

    basic_info = DBAPI.CustomizeFactory(kd_engine).custom(
        name=name,
        clause_list=clause_list,
        columns=columns + ['tradeCommiUnit'])
    # basic_info = basic_info.sort_values(by='listDate',
    #                                     ascending=False).drop_duplicates(
    #                                         subset='contractObject',
    #                                         keep='first')
    pdb.set_trace()
    basic_info = basic_info[
        (basic_info['lastTradeDate'] >= begin_date.strftime("%Y-%m-%d"))
        & (basic_info['lastTradeDate'] <= end_date.strftime("%Y-%m-%d"))
        #&(basic_info['tradeCommiUnit'] == '%')
    ]
    return basic_info.rename(columns={
        'contractObject': 'code',
        'code': 'symbol'
    })


def fetch_local_market1(base_path,
                        begin_date,
                        end_date,
                        codes,
                        method='pcr',
                        keep_symbol=False):
    ### 查找主力合约
    adj_name = "{0}_cumfactor".format(method) if isinstance(
        method, str) else "pcr_cumfactor"
    algin_factors = RetrievalAPI.get_algin_factors(
        begin_date=begin_date,
        end_date=end_date,
        codes=codes,
        columns=['trade_date', 'code', 'symbol', adj_name])
    algin_factors = algin_factors.drop_duplicates(
        subset=['trade_date', 'code'], keep='last')
    adj_columns = ['open', 'high', 'low', 'close', 'vwap']
    min_data = []
    for row in algin_factors.itertuples():
        # SF703_20260518.feather
        filename = os.path.join(
            base_path, row.trade_date.strftime("%Y%m%d"),
            "{0}_{1}.feather".format(row.symbol,
                                     row.trade_date.strftime("%Y%m%d")))
        md = pd.read_feather(filename)
        md['code'] = row.code
        if isinstance(method, str):
            md['factor'] = row.pcr_cumfactor
            md[adj_columns] = md[adj_columns].multiply(md['factor'], axis=0)
        min_data.append(md)
    min_data = pd.concat(min_data).sort_values(
        by=['datetime', 'code', 'symbol'])
    drop_colmns = ['vt_symbol', 'exchange', 'date', 'time']
    if not keep_symbol:
        drop_colmns.append('symbol')
    if isinstance(method, str):
        drop_colmns.append('factor')
    min_data = min_data.rename(columns={
        'datetime': 'trade_time',
        'open_interest': 'openint'
    }).drop(drop_colmns, axis=1)
    min_data['trade_time'] = pd.to_datetime(min_data['trade_time'])
    ## 过滤非正常交易时间段
    # times_to_exclude = [
    #     datetime.time(8, 59, 0),
    #     datetime.time(20, 59, 0),
    #     datetime.time(15, 16, 0)
    # ]

    t = min_data["trade_time"].dt.time
    day_session = (((t >= datetime.time(9, 0)) & (t <= datetime.time(11, 30)))
                   | ((t >= datetime.time(13, 30)) &
                      (t <= datetime.time(15, 0))))

    night_session = (((t >= datetime.time(21, 0)) &
                      (t <= datetime.time(23, 59, 59))) |
                     ((t >= datetime.time(0, 0)) &
                      (t <= datetime.time(2, 30))))

    # min_data = min_data[~min_data['trade_time'].dt.time.isin(times_to_exclude)]
    min_data = min_data[day_session | night_session]
    min_data['vwap'] = np.where(min_data['volume'] != 0, min_data['vwap'],
                                np.nan)
    min_data = min_data.sort_values(by=['trade_time', 'code', 'volume'])
    min_data['vwap'] = min_data['vwap'].ffill()
    min_data_cleaned = min_data[(min_data['volume'] != 0)
                                & (min_data['value'] != 0)].copy()
    return min_data_cleaned.reset_index(drop=True)


def fetch_main_market(begin_date,
                      end_date,
                      codes,
                      method='pcr',
                      keep_symbol=False,
                      forced_alignment=False):
    basic_info = fetch_basic(begin_date, end_date, codes)
    data = RetrievalAPI.get_main_price(begin_date=begin_date,
                                       end_date=end_date,
                                       codes=codes,
                                       method=method,
                                       format_data=0)
    res = []
    for code in data.keys():
        dt = data[code]
        dt['trade_time'] = pd.to_datetime(dt['barTime'])
        dt.rename(columns={
            'closePrice': 'close',
            'lowPrice': 'low',
            'highPrice': 'high',
            'openPrice': 'open',
            'totalVolume': 'volume',
            'totalValue': 'value',
            'openInterest': 'openint',
            'logRet': 'chg'
        },
                  inplace=True)

        dt = dt.drop(columns=['barTime', 'mincount', 'trade_date'], axis=1)
        if not keep_symbol:
            dt = dt.drop(columns=['symbol'], axis=1)

        #dt['price'] = dt['value'] / dt[
        #    'volume']  #此处用于成交价，但会出现value volume为0情况，导致price为inf，此情况使用 olch均值代替
        #dt['price'] = dt['price'].where(
        #    dt['price'].notna(), dt[['high', 'low', 'close',
        #                             'open']].mean(axis=1))
        dt['price'] = dt[['high', 'low', 'close', 'open']].mean(axis=1)
        #dt['vwap'] = dt['value'] / dt['volume']  ## 除以最小单位
        res.append(dt)
    data = pd.concat(res, axis=0)
    ## 临时 过滤重复数据
    data = data.merge(basic_info, on='code', how='left')
    data['vwap'] = data['value'] / data['volume'] / data['contMultNum']
    data = data.dropna(subset=['vwap'])
    data = data.drop_duplicates(subset=['trade_time', 'code']).sort_values(
        by=['trade_time', 'code'])

    if forced_alignment:
        data['trade_time'] = data['trade_time'] - pd.Timedelta(minutes=1)
    return data


def fetch_trader_market1(begin_time, end_time, code, adjusted_method='pcr'):
    mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
    dt1 = RetrievalAPI.get_algin_factors(begin_date=begin_time,
                                         end_date=end_time,
                                         codes=[code])

    if dt1.empty:
        print(f"[WARN] 未找到 {code} 的主力合约日历数据。")
        return pd.DataFrame()

    dt1['trade_date'] = pd.to_datetime(dt1['trade_date'])
    is_new_contract = (dt1['symbol'] != dt1['symbol'].shift()).cumsum()
    contract_blocks = dt1.groupby(is_new_contract).agg(
        symbol=('symbol', 'first'),
        start_date=('trade_date', 'min'),
        end_date=('trade_date', 'max')).reset_index(drop=True)
    all_chunks = []  # 用于收集查出来的每一段数据
    for row in contract_blocks.itertuples():
        print(row)
        current_symbol = row.symbol
        chunk_start_str = row.start_date.strftime('%Y-%m-%d 00:00:00')
        chunk_end_str = row.end_date.strftime('%Y-%m-%d 23:59:59')
        query = {
            "symbol": current_symbol,
            "datetime": {
                "$gte": chunk_start_str,
                "$lte": chunk_end_str
            }
        }
        cursor = mongo_client[
            os.environ['MG_COLL']]['impluse_market_bar'].find(query)
        chunk_df = pd.DataFrame(list(cursor))
        if chunk_df.empty:
            continue
        chunk_df['code'] = code
        chunk_df = chunk_df.drop(
            ['_id', 'vt_symbol', 'exchange', 'date', 'time'],
            axis=1).rename(columns={
                'open_interest': 'openint',
                'datetime': 'trade_time'
            })
        inter_columns = ['open', 'high', 'low', 'close']
        if isinstance(adjusted_method, str):
            factors = dt1[(dt1['trade_date'] >= chunk_start_str)
                          & (dt1['trade_date'] <= chunk_end_str) &
                          (dt1['symbol'] == current_symbol)]
            chunk_df['trade_date'] = pd.to_datetime(
                pd.to_datetime(chunk_df['trade_time']).dt.strftime('%Y-%m-%d'))
            chunk_df = pd.merge(chunk_df,
                                factors,
                                on=['trade_date', 'code', 'symbol'],
                                how='left')
            chunk_df[inter_columns] = chunk_df[inter_columns].multiply(
                chunk_df["{0}_cumfactor".format(adjusted_method)], axis=0)

        all_chunks.append(chunk_df)
    final_data = pd.concat(all_chunks, axis=0, ignore_index=True)
    final_data['trade_time'] = pd.to_datetime(final_data['trade_time'])
    return final_data


# def update_data(mongo_client, data, table_name):
#     insert_request = [
#         InsertOne(data) for data in data.to_dict(orient='records')
#     ]
#     delete_request = [
#         DeleteOne(data)
#         for data in data[['trade_time', 'code', 'symbol', 'task_id']].to_dict(
#             orient='records')
#     ]
#     _ = mongo_client['neutron'][table_name].bulk_write(
#         delete_request + insert_request, bypass_document_validation=True)


def fetch_metrics(category,
                  code,
                  names,
                  begin_time,
                  end_time,
                  table_name,
                  mongo_client=None):
    mongo_client = MongoDBManager(
        uri=os.environ['MG_URI']) if mongo_client is None else mongo_client
    query = {
        "code": code,
        "trade_time": {
            "$gte": begin_time,
            "$lte": end_time
        }
    }

    t_category = category if isinstance(category, list) else [category]
    query['category'] = {"$in": t_category}

    if isinstance(names, list) or isinstance(names, str):
        t_name = names if isinstance(names, list) else [names]
        query['name'] = {"$in": t_name}

    cursor = mongo_client[os.environ['MG_COLL']]["realm_{0}".format(
        table_name)].find(query)
    results = pd.DataFrame(list(cursor))
    results = results.drop(
        ['_id'], axis=1) if not results.empty else results.sort_values(
            by=['trade_time'])
    return results


def fetch_netout(category,
                 code,
                 begin_time,
                 end_time,
                 table_name,
                 mongo_client=None):

    mongo_client = MongoDBManager(
        uri=os.environ['MG_URI']) if mongo_client is None else mongo_client
    query = {
        "code": code,
        "trade_time": {
            "$gte": begin_time,
            "$lte": end_time
        }
    }

    t_category = category if isinstance(category, list) else [category]
    query['category'] = {"$in": t_category}
    print(query)
    cursor = mongo_client[os.environ['MG_COLL']]["realm_{0}".format(
        table_name)].find(query)

    results = pd.DataFrame(list(cursor))
    results = results.drop(['_id'], axis=1) if not results.empty else results
    return results
