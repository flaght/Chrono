import pdb
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from jdw import DBAPI
from alphacopilot.api.data import RetrievalAPI, ddb_tools, DDBAPI

name = "fut_tick"
cusomize_api = DDBAPI.cusomize_api()



clause_list1 = ddb_tools.to_format('Trade_Date', '==', ddb_tools.convert_date('2020-01-02'))
#clause_list1 = ddb_tools.to_format('Symbol', 'in', ['IC2002'])
#clause_list2 = ddb_tools.to_format('Volume', '<=',6)
pdb.set_trace()
df = cusomize_api.custom(table=name,
    clause_list=[clause_list1],
    format_data=1) # format 0: return DataFrame 1: return dict
print('--->')
df