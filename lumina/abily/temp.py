import lightgbm as lgb
import numpy as np
import pdb,time

choices = [-1, 0, 1]

size = 1071780
# 定义数组形状
shape = (size, 20)

X = np.random.choice(choices, size=shape)
# X = np.random.rand(107178, 20)
y = np.random.randint(0, 3, size=size)
pdb.set_trace()
#print(y)
train_data = lgb.Dataset(X, label=y)


params = {
    #'boosting_type': 'gbdt',
    #'objective': 'binary',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 4,  # 对于弱信号，简单的树可能更好
    'n_jobs': -1,
    #'device': 'gpu',  # 关键参数！
    'verbose': 1
}

start1 = time.time()

print("开始训练...")
bst = lgb.train(params, train_data, num_boost_round=10)
print("训练完成")
print("{0}".format(time.time() - start1))


