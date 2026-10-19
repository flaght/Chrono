import os, pdb
from datetime import datetime
import pandas as pd
from pathlib import Path
from kdutils.macro import base_path
from lib.svx001 import scale_factors
from lib.macro import REGIME_FEATURES, PREDICT_FEATURES, TEXT_FEATURES


def process_regime_data(begin_date, end_date):
    base_dir = os.path.join(base_path, "live", "temp", end_date, "basic")
    regime_data = pd.read_feather(os.path.join(base_dir,
                                               "regime_data.feather"))
    regime_data = regime_data[(regime_data['trade_date'] >= begin_date)
                              & (regime_data['trade_date'] <= end_date)]
    regime_data = regime_data.dropna().reset_index(drop=True)
    regime_data = regime_data[['trade_date'] + REGIME_FEATURES]
    return regime_data


def process_predict_data(begin_date, end_date):
    base_dir = os.path.join(base_path, "live", "temp", end_date, "basic")
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


def process_textuals(begin_date, end_date, premarket=True):
    textuals_dirs = os.path.join(base_path, "data", "event", "textuals")
    file_path = Path(textuals_dirs)
    res = []
    end_date1 = end_date if not premarket else advanceDateByCalendar(
        'china.sse', end_date, '1b').strftime('%Y-%m-%d')
    for feat_file in file_path.rglob('*.feather'):
        if begin_date <= feat_file.stem <= end_date1:
            ed = pd.read_feather(feat_file)
            ed['date'] = feat_file.stem
            res.append(ed)
    event_data = pd.concat(res, axis=0).sort_values(by=['date'])
    event_data = event_data.rename(columns={
        'date': 'trade_date'
    }).reset_index(drop=True)
    if premarket:
        event_data['trade_date'] = end_date
    event_data['trade_date'] = pd.to_datetime(event_data['trade_date'])
    return event_data[event_data['feature_type'].isin(
        TEXT_FEATURES)].reset_index(drop=True)


if __name__ == '__main__':
    from ultron.tradingday import *
    begin_date = advanceDateByCalendar('china.sse', datetime.now(),
                                       '-1b').strftime('%Y-%m-%d')
    end_date = begin_date

    dates = makeSchedule(begin_date, end_date, '1b', 'china.sse',
                         BizDayConventions.Preceding)

    for date in dates:
        print(date)
        begin_date = end_date = date.strftime('%Y-%m-%d')
        textuals_data = process_textuals(begin_date=begin_date,
                                         end_date=end_date)
        predict_data = process_predict_data(begin_date=begin_date,
                                            end_date=end_date)
        regime_data = process_regime_data(begin_date=begin_date,
                                          end_date=end_date)

        output_dir = os.path.join(base_path, "live", "temp", end_date,
                                  "normal")
        os.makedirs(output_dir, exist_ok=True)
        pdb.set_trace()
        regime_data.to_feather(os.path.join(output_dir, "regime_data.feather"))
        predict_data.to_feather(
            os.path.join(output_dir, "predict_data.feather"))
        textuals_data.to_feather(
            os.path.join(output_dir, "textuals_data.feather"))
