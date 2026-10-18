import pdb
from dotenv import load_dotenv

load_dotenv()

from jdw import DBAPI

kd_engine = DBAPI.FetchEngine.create_engine('kd')


def fetch_basic(codes, columns=['contractObject', 'contMultNum', 'listDate']):
    name = 'fut_basic'
    names = DBAPI.CustomizeFactory(kd_engine).name(name=name)
    if isinstance(codes, list):
        clause_list = [names.contractObject.in_(codes), names.flag == 1]
    elif isinstance(codes, str):
        clause_list = [names.contractObject.in_([codes]), names.flag == 1]
    elif codes is None:
        clause_list = [names.flag == 1]

    basic_info = DBAPI.CustomizeFactory(kd_engine).custom(
        name=name, clause_list=clause_list, columns=columns)
    basic_info = basic_info.sort_values(by='listDate',
                                        ascending=False)
    pdb.set_trace()
    return basic_info.rename(columns={
        'code': 'symbol',
        'contractObject': 'code'
    })


def fetch_algin_factors(begin_date, end_date, codes=None, columns=None):
    name = 'fut_algin_factors'
    names = DBAPI.CustomizeFactory(kd_engine).name(name=name)
    clause_list = [
        names.trade_date >= begin_date, names.trade_date <= end_date
    ]
    if isinstance(codes, list):
        clause_list.append(names.code.in_(codes))
    algin_factors_data = DBAPI.CustomizeFactory(kd_engine).custom(
        name=name, clause_list=clause_list, columns=columns)
    return algin_factors_data


def test1():

    # fetch_algin_factors(begin_date='2026-01-01',
    #                     end_date='2026-05-01',
    #                     codes=['RB', 'T'],
    #                     columns=None)

    # dt1 = fetch_basic(codes=['RB', 'T'])
    # print(dt1)

    dt2 = fetch_basic(
        codes=None,
        columns=[
            'contractObject', 'listDate', 'secFullName', 'secShortName',
            'exchangeCD', 'contractType', 'code', 'minChgPriceNum',
            'minChgPriceUnit', 'priceValidDecimal', 'limitUpNum',
            'limitDownNum', 'contMultNum', 'contMultUnit', 'tradeMarginRatio',
            'lastTradeDate', 'firstDeliDate', 'lastDeliDate', 'deliMethod',
            'tradeCommiNum', 'tradeCommiUnit', 'deliCommiNum', 'deliCommiUnit',
            'listBasisPrice', 'prodID'
        ])
    pdb.set_trace()
    print(dt2)


test1()
