import os, pdb, sys, json, math, empyrical, argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv

load_dotenv()
from ultron.factor.genetic.geneticist.operators import *
from kdutils.common import fetch_temp_data, fetch_temp_returns
from lib.cux002 import *
from lib.cux003 import generate_simple_id, create_id

expressions = {
    "MDPO(4,'tc005_1_1_2_1')":
    -1,
    "MADiff(2,'ixy007_1_2_1')":
    -1,
    "SIGMOID(MADiff(3,'iv012_1_2_1'))":
    -1,
    "MRes(3,'tc015_1_2_1','tn003_1_1_2_3_1')":
    -1,
    "MCPS(2,'oi034_1_2_1')":
    1,
    "MRes(4,SIGN(SIGLOG10ABS(MMAX(3,MRes(4,'tn008_1_2_1_1',SIGLOG10ABS('tc001_2_3_0'))))),'tc014_1_1_2_0')":
    -1,
    "SIGMOID(MDPO(3,MADiff(3,SIGMOID(SIGMOID('iv012_1_2_1')))))":
    -1,
    "MDEMA(2,'tn003_2_1_2_3_1')":
    -1,
    "MADiff(4,SUBBED('tn008_1_2_1_4',MRes(4,'tn009_1_2_0_4','ixy007_1_2_1')))":
    1,
    "MDIFF(2,ADDED('ixy007_1_2_0','tv017_1_2_1'))":
    -1,
    "MRes(3,'tc015_1_2_1',EMA(3,'tn003_1_1_2_3_1'))":
    -1,
    "EMA(3,ABS(MPERCENT(3,'tc006_1_2_0')))":
    -1,
    "MT3(2,'ixy006_1_2_1')":
    -1,
    "MDIFF(2,'tn003_1_1_2_3_1')":
    -1,
    "MRes(4,SIGN(SIGLOG10ABS('tc001_2_3_0')),'tc014_1_1_2_0')":
    -1,
    "EMA(2,MCPS(2,'oi034_1_2_1'))":
    1,
    "MDPO(3,'tv017_1_2_1')":
    -1,
    "MMeanRes(4,'dv005_1_2_1','ixy008_1_2_1')":
    -1
}


def create_factor(expression, total_data):
    factor_data1 = calc_factor(expression=expression,
                               total_data=total_data.copy(),
                               indexs=[],
                               key='code')
    backup_cycle = 1
    factor_data1 = factor_data1.replace([np.inf, -np.inf], np.nan)
    factor_data1['transformed'] = np.where(
        np.abs(factor_data1.transformed.values) > 0.000001,
        factor_data1.transformed.values, np.nan)
    factor_data1 = factor_data1.loc[factor_data1.index.unique()[backup_cycle:]]
    return factor_data1


def perf(factor_data1, total_data, period, expression):
    dt1 = factor_data1.reset_index().merge(total_data.reset_index()[[
        'trade_time', 'code', 'nxt1_ret_{0}h'.format(period)
    ]],
                                           on=['trade_time', 'code'])
    is_on_mark = dt1['trade_time'].dt.minute % int(period) == 0
    dt1 = dt1[is_on_mark]
    evaluate1 = FactorEvaluate1(factor_data=dt1,
                                factor_name='transformed',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=240,
                                fee=0.000,
                                scale_method='roll_zscore',
                                expression=expression)
    stats_df = evaluate1.run()
    stats_df['expression'] = expression
    stats_df['name'] = create_id(generate_simple_id(expression))
    return evaluate1.factor_data['f_scaled'], stats_df


### 创建因子
def create_factors(method, symbol, period):
    pdb.set_trace()
    alpha_res = []
    state_res = []
    total_data = fetch_data(method=method, instruments=symbol).set_index(
        ['trade_time', 'code']).unstack()
    #total_data = total_data.fillna(
    #    method='ffill').stack().reset_index().set_index('trade_time')
    i = 0
    pdb.set_trace()
    for expression, direction in expressions.items():
        print(expression)
        i += 1
        
        factor_data1 = create_factor(expression=expression,
                                     total_data=total_data.copy())
        ff1 = factor_data1.shape[0]
        ff2 = factor_data1.dropna().shape[0]
        if ff1 != ff2:
            pdb.set_trace()
            print('')
        #factor_data1['transformed'] = factor_data1['transformed'].fillna(method='ffill')
        ### 校验因子
        factors_data2, stats = perf(factor_data1=factor_data1,
                                    total_data=total_data.copy(),
                                    period=period,
                                    expression=expression)
        print(expression, stats)
        factors_data2.name = stats['name']
        alpha_res.append(factors_data2)
        state_res.append(stats)

    final_factors = pd.concat(alpha_res, axis=1)
    final_state = pd.DataFrame(state_res)
    outdirs = os.path.join("records", "temp2", method, symbol, str(period))
    if not os.path.exists(outdirs):
        os.makedirs(outdirs)
    final_factors = final_factors.reset_index()
    final_factors['code'] = 'IM'
    returns_data = total_data[['code', 'nxt1_ret_15h']].reset_index()
    final_factors = final_factors.merge(returns_data,
                                        on=['trade_time', 'code'])
    final_factors = final_factors.sort_values(by=['trade_time', 'code'],
                                              ascending=True)
    final_factors.to_feather(
        os.path.join(outdirs,
                     "final_factors_{0}_{1}.feather".format(method, period)))
    final_state.to_csv(os.path.join(
        outdirs, "final_state_{0}_{1}.csv".format(method, period)),
                       encoding="UTF-8")


### 训练模型
def train_model(method, symbol, period):
    ## 读取数据
    random_state = 42
    pdb.set_trace()
    outdirs = os.path.join("records", "temp2", method, symbol, str(period))
    final_factors = pd.read_feather(
        os.path.join(outdirs,
                     "final_factors_{0}_{1}.feather".format(method, period)))

    final_factors1 = pd.read_feather(
        os.path.join(
            os.path.join("records", "temp2", 'bicso1', symbol, str(period)),
            "final_factors_{0}_{1}.feather".format('bicso1', period)))
    final_factors2 = pd.concat(
        [final_factors.dropna(),
         final_factors1.dropna()], axis=0)
    final_factors = final_factors2.set_index(
        'trade_time').loc['2022-08-12':'2023-12-31'].reset_index()
    final_factors1 = final_factors2.set_index(
        'trade_time').loc['2024-01-01':].reset_index()
    final_factors = final_factors.set_index(['trade_time', 'code']).dropna()
    final_factors1 = final_factors1.set_index(['trade_time', 'code']).dropna()
    features = [
        col for col in final_factors.columns if col not in ['nxt1_ret_15h']
    ]
    X_train_scaled = final_factors[features]
    y_transformed = final_factors['nxt1_ret_15h']
    X_train, X_val, y_train, y_val = train_test_split(X_train_scaled,
                                                      y_transformed,
                                                      test_size=0.2,
                                                      shuffle=False)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
    params = {
        #'objective': 'regression_l1',
        'metric': 'mae',  # Mean Absolute Error
        'n_estimators': 200,  # 可以设置一个较大的值，由早停来控制
        'learning_rate': 0.05,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 1,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'num_leaves': 31,
        'min_gain_to_split': 0.0,  # 确保任何非负增益都会分裂
        'min_child_samples': 5,  # 大幅减小叶子节点的最小样本数，允许更深的分裂
        'min_child_weight': 1e-5,  # 同样减小
        'verbose': -1,
        'n_jobs': -1,
        'seed': random_state,
        'boosting_type': 'gbdt',
        #'device': 'gpu',
        #'gpu_platform_id': 0,
        #'gpu_device_id': 0,
        'verbose': 1
    }
    pdb.set_trace()
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(10, verbose=True)]  # 早停是关键  
    )

    feature_importances = pd.Series(model.feature_importance(), index=features)
    raw_meta = model.predict(final_factors1[features],
                             num_iteration=model.best_iteration)
    factors_data1 = pd.DataFrame(raw_meta,
                                 index=final_factors1.index,
                                 columns=['predict'])
    factors_data1.reset_index().to_feather("222.feather")
    pdb.set_trace()
    print('-->')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--dethod', type=str, default='ic', help='data method')

    parser.add_argument('--method',
                        type=str,
                        default='bicso1',
                        help='data method')
    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instrument')

    parser.add_argument('--task_id',
                        type=str,
                        default='200036',
                        help='code or instruments')

    parser.add_argument('--period',
                        type=str,
                        default='15',
                        help='code or instruments')

    args = parser.parse_args()
    create_factors(method=args.method,
                   symbol=args.instruments,
                   period=args.period)
