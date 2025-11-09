from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix

from lib.syn001 import build_factors, linear_train_model, lassocv_train_model, rigde_train_model, lgb_train_model
#from lib.syn002 import lgb_optuna_model
#from lib.syn003 import multi_lgb_optuna_model

if __name__ == '__main__':
    variant = Tactix().start()

    if variant.form == 'build':
        build_factors(method=variant.method,
                      instruments=variant.instruments,
                      task_id=variant.task_id,
                      period=variant.period)

    elif variant.form == 'linear':
        linear_train_model(method=variant.method,
                           instruments=variant.instruments,
                           task_id=variant.task_id,
                           period=variant.period)

    elif variant.form == 'lassocv':
        lassocv_train_model(method=variant.method,
                            instruments=variant.instruments,
                            task_id=variant.task_id,
                            period=variant.period)

    elif variant.form == 'rigde':
        rigde_train_model(method=variant.method,
                          instruments=variant.instruments,
                          task_id=variant.task_id,
                          period=variant.period)

    elif variant.form == 'lgb':
        lgb_train_model(method=variant.method,
                        instruments=variant.instruments,
                        task_id=variant.task_id,
                        period=variant.period)

    elif variant.form == 'opt_lgb':
        lgb_optuna_model(method=variant.method,
                         instruments=variant.instruments,
                         task_id=variant.task_id,
                         period=variant.period)
    
    elif variant.form == 'multi_opt_lgb':
        multi_lgb_optuna_model(
            method=variant.method,
                         instruments=variant.instruments,
                         task_id=variant.task_id,
                         period=variant.period
        )
