import os, sys, pdb, re, math, json, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import ultron.factor.empyrical as empyrical
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
import sqlalchemy.orm as orm

load_dotenv()

sys.path.insert(0, os.path.abspath('../'))

from lumina.genetic import Motor
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_pnl

two_operators_sets = ['MCORR', 'MUL', 'MRes']
one_operators_sets = ['MRANK', 'ACOS']


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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


def callback_models(gen, rootid, best_programs, custom_params):
    g_instruments = custom_params['g_instruments']
    dethod = custom_params['dethod']
    base_path = custom_params['base_path']
    tournament_size = custom_params['tournament_size']
    standard_score = custom_params['standard_score']
    best_programs = [program.output() for program in best_programs]
    best_programs = pd.DataFrame(best_programs)

    data_programs = best_programs.copy()
    data_programs['task_id'] = rootid
    data_programs['strategy_params'] = data_programs['strategy_params'].apply(
        lambda x: json.dumps(x, cls=NpEncoder))
    data_programs['signal_params'] = data_programs['signal_params'].apply(
        lambda x: json.dumps(x, cls=NpEncoder))

    data_adapter.refresh_data(total_data=data_programs.drop(
        ['update_time', 'features'], axis=1),
                              method='increment',
                              table_name='genetic_strategy')
    
    dirs = os.path.join(base_path, dethod, g_instruments, 'evolution')
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    #names = custom_params[rootid]
    filename = os.path.join(dirs, f'{rootid}.feather')
    if os.path.exists(filename):
        old_dt = pd.read_feather(filename)
        best_programs = pd.concat([old_dt, best_programs], axis=0)

    best_programs = best_programs.drop_duplicates(subset=['name'])
    final_programs = best_programs[best_programs['fitness'] > standard_score]
    if final_programs.shape[0] < tournament_size:
        best_programs = best_programs.sort_values('fitness', ascending=False)
        final_programs = best_programs.head(tournament_size)
    final_programs.sort_values(
        'fitness', ascending=False).reset_index(drop=True).to_feather(filename)


def callback_fitness(factor_data, total_data, signal_method, strategy_method,
                     factor_sets, custom_params, default_value):

    strategy_settings = {}
    factor_data = factor_data.reset_index().set_index(['trade_time', 'code'])
    total_data = total_data.set_index(['trade_time', 'code']).unstack()
    pos_data = signal_method.function(factor_data=factor_data,
                                      **signal_method.params)
    pos_data = strategy_method.function(signal=pos_data,
                                        total_data=total_data,
                                        **strategy_method.params)
    pdb.set_trace()
    df = calculate_ful_ts_pnl(pos_data=pos_data,
                              total_data=total_data,
                              strategy_settings=strategy_settings)
    ### 值有异常 绝对值大于1
    returns = df['ret']
    fitness = empyrical.sharpe_ratio(returns=returns, period=empyrical.DAILY)
    return fitness


def evolution(rootid, method):
    filename = os.path.join('records', method, 'IF', 'factors',
                            "factors_data.feather")
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])

    factors_data = factors_data.set_index('trade_time')
    factor_columns = [
        col for col in factors_data.columns if col not in [
            'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
            'volume', 'openint', 'vwap'
        ]
    ]
    operators_sets = two_operators_sets + one_operators_sets

    configure = {
        'n_jobs': 1,
        'population_size': 32,
        'tournament_size': 8,
        'init_depth': 3,
        'rootid': '20250414',
        'custom_params': {
            'g_instruments': 'IF',
            'dethod': method,
            'base_path': os.path.join('records', method, 'IF', 'evolution'),
            'tournament_size': 5,
            'standard_score': 1
        }
    }
    motor = Motor(factor_columns=factor_columns,
                  callback_fitness=callback_fitness,
                  callback_save_model=callback_models)
    pdb.set_trace()
    motor.calculate(total_data=factors_data,
                    configure=configure,
                    operators_sets=operators_sets,
                    signals_sets=None,
                    strategies_sets=None)


evolution('11111', 'aa1')
