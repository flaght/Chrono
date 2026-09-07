### 市场特征 预测特征 滚动标准化
import os, pdb
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from kdutils.macro import base_path
from kdutils.ttimes import get_dates, get_ranges
from lib.svx001 import scale_factors
from macro import REGIME_FEATURES, PREDICT_FEATURES, TEXT_FEATURES


def process_regime_data(method):
    begin_date, end_date = get_dates(method=method)
    base_dir = os.path.join(base_path, "basic", method)
    regime_data = pd.read_feather(os.path.join(base_dir,
                                               "regime_data.feather"))
    # factor_names = [
    #     f for f in regime_data.columns
    #     if f not in ['trade_date', 'code', 'nxt1_ret_1h']
    # ]
    # for f in factor_names:
    #     print(f)
    #     scale_factors(
    #         factor_data=regime_data,
    #         method='roll_zscore',
    #         win=5,  # 放在环境变量里，原始数据扩展
    #         factor_name=f)
    #     regime_data[f] = regime_data['transformed']
    #     regime_data.drop(['transformed'], axis=1, inplace=True)
    regime_data = regime_data[(regime_data['trade_date'] >= begin_date)
                              & (regime_data['trade_date'] <= end_date)]
    regime_data = regime_data.dropna().reset_index(drop=True)
    regime_data = regime_data[['trade_date'] + REGIME_FEATURES]
    return regime_data


def process_textuals(method):
    begin_date, end_date = get_dates(method=method)
    textuals_dirs = os.path.join(base_path, "data", "event", "textuals")
    file_path = Path(textuals_dirs)
    res = []
    for feat_file in file_path.rglob('*.feather'):
        if begin_date <= feat_file.stem <= end_date:
            ed = pd.read_feather(feat_file)
            ed['date'] = feat_file.stem
            res.append(ed)
    event_data = pd.concat(res, axis=0).sort_values(by=['date'])
    event_data = event_data.rename(columns={
        'date': 'trade_date'
    }).reset_index(drop=True)
    event_data['trade_date'] = pd.to_datetime(event_data['trade_date'])
    return event_data[event_data['feature_type'].isin(
        TEXT_FEATURES)].reset_index(drop=True)


def process_predict_data(method):
    begin_date, end_date = get_dates(method=method)
    base_dir = os.path.join(base_path, "basic", method)
    predict_data = pd.read_feather(
        os.path.join(base_dir, "predict_data.feather"))
    factor_names = [
        f for f in predict_data.columns
        if f not in ['trade_date', 'code', 'nxt1_ret_1h']
    ]
    for f in factor_names:
        scale_factors(
            factor_data=predict_data,
            method='roll_zscore',
            win=5,  # 放在环境变量里，原始数据扩展
            factor_name=f)
        predict_data[f] = predict_data['transformed']
        predict_data.drop(['transformed'], axis=1, inplace=True)
    predict_data = predict_data[(predict_data['trade_date'] >= begin_date)
                                & (predict_data['trade_date'] <= end_date)]
    predict_data = predict_data.dropna().reset_index(drop=True)
    return predict_data[['trade_date'] + PREDICT_FEATURES]


def process_returns(method, period):
    begin_date, end_date = get_dates(method=method)
    base_dir = os.path.join(base_path, "basic", method)
    return_data = pd.read_feather(os.path.join(base_dir,
                                               "return_data.feather"))
    return_data = return_data[(return_data['trade_date'] >= begin_date)
                              & (return_data['trade_date'] <= end_date)]
    return_data = return_data[[
        'trade_date', 'code', "nxt1_ret_{0}h".format(period)
    ]]
    return return_data


def splite_data(data, train_range, val_range, test_range):
    train_data = data[(data['trade_date'] >= train_range[0])
                      & (data['trade_date'] <= train_range[1])]
    val_data = data[(data['trade_date'] >= val_range[0])
                    & (data['trade_date'] <= val_range[1])]

    test_data = data[(data['trade_date'] >= test_range[0])
                     & (data['trade_date'] <= test_range[1])]
    return train_data.reset_index(drop=True), val_data.reset_index(
        drop=True), test_data.reset_index(drop=True)


## 输出切割
if __name__ == '__main__':
    method = 'test0'
    period = 3
    regime_data = process_regime_data(method=method)
    predict_data = process_predict_data(method=method)
    textuals_data = process_textuals(method=method)
    returns_data = process_returns(method=method, period=period)
    
    # aligned_data = pd.merge(returns_data,textuals_data,on='trade_date',how='left') 主要用于检查 textuals_data是否有缺收
    output_dir = os.path.join(base_path, "normal", method)
    os.makedirs(output_dir, exist_ok=True)
    train_range, val_range, test_range = get_ranges(method=method)
    train_regime_data, val_regime_data, test_regime_data = splite_data(
        data=regime_data,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range)

    train_predict_data, val_predict_data, test_predict_data = splite_data(
        data=predict_data,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range)

    train_textuals_data, val_textuals_data, test_textuals_data = splite_data(
        data=textuals_data,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range)

    train_returns_data, val_returns_data, test_returns_data = splite_data(
        data=returns_data,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range)
    pdb.set_trace()
    train_regime_data.to_feather(
        os.path.join(output_dir, "train_regime_data.feather"))
    train_predict_data.to_feather(
        os.path.join(output_dir, "train_predict_data.feather"))
    train_textuals_data.to_feather(
        os.path.join(output_dir, "train_textuals_data.feather"))
    train_returns_data.to_feather(
        os.path.join(output_dir, "train_returns_data.feather"))

    val_regime_data.to_feather(
        os.path.join(output_dir, "val_regime_data.feather"))
    val_predict_data.to_feather(
        os.path.join(output_dir, "val_predict_data.feather"))
    val_textuals_data.to_feather(
        os.path.join(output_dir, "val_textuals_data.feather"))
    val_returns_data.to_feather(
        os.path.join(output_dir, "val_returns_data.feather"))

    test_regime_data.to_feather(
        os.path.join(output_dir, "test_regime_data.feather"))
    test_predict_data.to_feather(
        os.path.join(output_dir, "test_predict_data.feather"))
    test_textuals_data.to_feather(
        os.path.join(output_dir, "test_textuals_data.feather"))
    test_returns_data.to_feather(
        os.path.join(output_dir, "test_returns_data.feather"))

    regime_data.to_feather(os.path.join(output_dir, "regime_data.feather"))
    predict_data.to_feather(os.path.join(output_dir, "predict_data.feather"))
    textuals_data.to_feather(os.path.join(output_dir, "textuals_data.feather"))
    returns_data.to_feather(os.path.join(output_dir, "returns_data.feather"))
