from lib.syn001.trainer import Trainer
from lib import logger

def train_model(model_class, train_data, test_data, selected_features, 
            params, roll_win, period, outdirs):
    trainer = Trainer(params=params)
    code = train_data['code'].unique().tolist()[0]

    X, y, dates = trainer.prepare_data(df=train_data, selected_features=selected_features,
                taget_col="nxt1_ret_{0}h".format(period))

    X_train, X_val, y_train, y_val, dates_train, dates_val = trainer.split_data(
        X, y, dates, train_ratio=0.7)

    model = trainer.train_single(model_class=model_class,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    selected_features=selected_features)
    
    y_train_pred = trainer.predict(X_train, model)
    
    ## 训练集
    trainer.predict_all(X=X_train, dates_val=dates_train,
                code=code, period=period,roll_win=roll_win,
                data=train_data, expression='train',outdirs=outdirs)
    ## 校验集
    trainer.predict_all(X=X_val, dates_val=dates_val,
                code=code, period=period,roll_win=roll_win,
                data=train_data, expression='val',outdirs=outdirs)

    ##测试集
    X_test, y_test, date_test = trainer.prepare_data(
        test_data, selected_features, "nxt1_ret_{}h".format(period))

    trainer.predict_all(X=X_test, dates_val=date_test,
                code=code, period=period,roll_win=roll_win,
                data=test_data, expression='test',outdirs=outdirs)