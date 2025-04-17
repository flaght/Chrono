import pdb
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
import sqlalchemy as sa

def fetch_factors(task_id):
    engine = sa.create_engine("mysql+mysqlconnector://neutron:123456@10.63.6.155:12306/quant") 
    sql_str = "select name, formual, fitness from genetic_factors where task_id = '{0}' and fitness > 2 order by fitness desc limit 400;".format(task_id)
    data = pd.read_sql(sql=sql_str, con=engine)
    print(data)
    return data

horizon = 5
mapping = {1:'24121217',3:'24121408',5:'24121409'}
data = fetch_factors(mapping[horizon])
pdb.set_trace()
data.to_csv("sqta_{0}.csv".format(horizon),encoding='UTF-8')
print('-->')