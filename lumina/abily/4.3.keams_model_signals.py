import os, pdb
import pandas as pd
from lumina.genetic.fusion import Rotor
from lumina.genetic import Actuator
from kdutils.cache import exist_cache, load_cache 

code = 'IM'
task_id = '100013'
method = 'aicso2'

positions_data = load_cache(code=code,
                            task_id=task_id,
                            method=method,
                            cache_file='positions_data.pkl')
positions_data.name = 'value'
positions_data = positions_data.reset_index().set_index('trade_time')
positions_data.head()

code = 'im'
def fetch_rotor(name):
    base_path = '/workspace/worker/pj/Chrono/lumina/abily/records/aicso2/ims/kmeans'
    pdb.set_trace()
    rotor = Rotor.from_pickle(
            path=os.path.join(base_path, code.lower()), name=name)
    return rotor

rotor1 = fetch_rotor('1034416114')
signal_data = rotor1.predict(positions_data)
signal_data.name = 'signal'
t1_dt = pd.concat([positions_data, signal_data], axis=1)
short = t1_dt[t1_dt['signal']==-1].value.max()
long = t1_dt[t1_dt['signal']==1].value.max()
zero = t1_dt[t1_dt['signal']==0].value.max()
print("short:{0}, long:{1}, zero:{2}, mapping:{3}".format(short, long, zero, rotor1._best_mapping))