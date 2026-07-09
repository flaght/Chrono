### 筛选因子特征标准
1. 研究标的 卡玛大于5， IC绝对值大于0.02， 夏普大于1.5
2. 对比标的 卡玛大于3， IC绝对值大于0.02， 夏普大于1.3 


### 强化学习特征
1. 选择强逻辑性的特征，除了满足绩效要求，还要满足逻辑性要求
2. 放入到强化学习模型里，采用滚动方式训练
3. 对于整体的绩效的增加


### IM 记录：
1053150819189592

### 流程

1.1.1.create_basic_factors 
- 1. factors  merge 构建基础字段
- 2. returns 创建收益率
- 3. 切割数据 train val test

2.1.0.metrics_basic_factors
- 1. 评估基础字段绩效


2.1.2.gentic_motor_factors.py：
- 1. 进化算法挖掘因子

2.1.3.directed_motor_factors.py
- 1. 进化算法定向挖掘

2.1.4.optuna_parellel_factors.py
2.1.5.optuna_parellel_factors.py
- 1. 贝叶斯寻优定向挖掘

2.1.6.optuna_parellel_factors.py
- 2. 贝叶斯寻优挖掘因子

2.2.1.scope_valid_factors.py
- 1. 计算指定筛选挖掘因子绩效生成图
- 2. 计算对比品种筛选挖掘因子绩效生成图

2.2.2.compare_twin_factors
- 1. 对比双品种绩效对比生成图

2.2.3.choose_deform_factors
- 1. 生成指定品种生成图




2.2.6.model_select_factor.py
- 1. 选中线性模型或者sklearn模型训练

2.2.7.create_metrics.py
- 1. 选中因子生成绩效文件

3.0.1.preprocess_data.py
- 1. 因子调整方向，时序标准化
- 2. 强化学习模型训练准备数据

3.0.2.blend_factors.py
- 1. 筛选因子 构建等权组合

3.1.1.predict_synthesis.py
- 1. sklearn 模型预测结果

3.1.2.temporal_convolution.py
- 1. TCN 预测

3.1.3.lgbm_synthesis.py
- 1. lgbm合成

3.1.4.linear_synthesis.py
- 选中线性模型或者sklearn模型训练


3.2.1.autoencoder.py
- autoencoder 编码重构

3.2.2.sequential.py 3.2.3.sequentialnll.py 3.2.4.seqdeconlynll.py
- NLL 模型


4.1.1.signal_metrics.py
- 预测的er值转换为信号并生成信号

4.1.1.signal_transform.py
- 预测的er值转换为信号并生成信号，多种信号函数

6.0.1.create_regime_data
- 计算对应的 regmie 因子


6.0.1.linear_increment_factors.py
- 指定因子等权合成后绩效评估

步骤说明：

- 1. 执行 1.1.1.create_basic_factors.py
    - 作用: 创建基础字段，创建不同周期收益率，切割数据集 训练集,校验集,测试集,近期数据集。 近期数据集用于看因子最近表现


- 2.1  执行 2.1.2.gentic_motor_factors.py
    - 作用: 根据 1.1.1 生成的基础字段，指定收益率  进化算法无定向挖掘因子
- 2.2 执行 2.1.3.directed_motor_factors.py
    - 作用: 根据 1.1.1 生成的基础字段，指定收益率  进化算法定向挖掘因子，所谓定向就是指定算子，指定特征进行挖掘
- 2.3 执行 2.1.5.optuna_parellel_factors.py
    - 作用: 根据 1.1.1 生成的基础字段，指定收益率  进化算法定向挖掘因子 指定表达式进行定向挖掘
- 2.4 执行 2.1.6.optuna_parellel_factors.py
    - 作用: 根据 1.1.1 生成的基础字段，指定收益率  进化算法定向挖掘因子 指定基础字段类进行定向挖掘

- 3.  执行 2.2.1.scope_valid_factors_parallel.py
    - 作用: 根据上述挖掘模块挖掘出来的因子，根据复合指标 IC, 卡玛 夏普初步筛选因子

- 4.  执行 2.2.3.choose_deform_factors.py
    - 作用 根据上述筛选出来的因子 在训练集和校验集 生成绩效图

- 5.  21.2.人工初筛绩效因子.ipynb
    - 作用: 人工筛选因子

- 6. 执行 2.2.3.choose_deform_factors.py
    - 作用: 根据上述筛选出来的因子 在近期集 生成绩效图

- 7. 21.3.人工再筛绩效因子.ipynb
    - 作用: 通过绩效图筛选近期也表现不错的因子

- 8.  执行 3.0.1.preprocess_data.py  build
    - 作用: 筛选的因子进行方向调整和时序标准化

- 9.  执行 3.0.1.preprocess_data.py  prepare
    - 作用: 切割因子数据，创建训练集 校验集 测试集

- 10. 执行  3.0.2.blend_factors.py  selected
    - 作用: 根据剔除高相关性因子


- 11.1.  执行 3.1.4.blend_synthesis.py  predict
    - 作用: 使用原生方法根据不同相关性筛选的因子进行等权合成, 绩效评估也在里面
- 11.2.  执行 3.1.4.blend_synthesis.py  forecast
    - 作用: 使用WF方法根据不同相关性筛选的因子进行等权合成, 绩效评估也在里面

对比两个合成方法结果是否一致


- 12.1 执行 3.1.5.blend_signal_backtest.py  build
    - 作用: 把原始方法合成后的er 转成 信号 

- 12.2 执行 3.1.5.blend_signal_backtest.py  metrics
    - 作用: 原始方法对信号进行绩效评估

- 12.3 执行 3.1.5.linear_signal_backtest.py  backtest
    - 作用: 原始方法信号转成交易规则进行回测

- 12.4 执行 3.1.5.linear_signal_backtest.py  wfs
    - 作用: 把WF方法合成后的er转成信号 

- 12.5 执行 3.1.5.linear_signal_backtest.py  wfm
    - 作用: WF方法对信号进行绩效评估

- 12.6 执行 3.1.5.linear_signal_backtest.py  wfb
    - 作用: WF方法转成交易规则进行回测


- 13.1  执行 6.1.3.build_rl_strategy.py train
    - 作用: 根据相关性筛选因子，进行强化学习模型训练

- 13.2  执行 6.1.3.build_rl_strategy.py predict
    - 作用: 使用模型预测训练集 校验集 测试集 
    
- 13.3  执行 6.1.3.build_rl_strategy.py eval
    - 作用: 对训练集 校验集 测试集 进行评估

- 13.4  执行 6.2.1.create_rl_strategy.py predict
    - 作用:  使用原生方法对模型生成校验集和测试集

- 13.5  执行 6.2.1.create_rl_strategy.py forecast
    - 作用: 使用WF方法对模型生成校验集和测试集

- 13.6  执行 6.2.1.create_rl_strategy.py metrics
    - 作用: 对 原始方式和WF方式生成校验集和测试集的绩效值

- 13.7 执行 6.2.2.rl_signal_backtest.py  build
    - 作用: 把原始方法合成后的er 转成 信号 

- 13.8 执行 6.2.2.rl_signal_backtest.py  metrics
    - 作用: 原始方法对信号进行绩效评估

- 13.9 执行 6.2.2.rl_signal_backtest.py  backtest
    - 作用: 原始方法信号转成交易规则进行回测

- 13.10 执行 6.2.2.rl_signal_backtest.py  wfs
    - 作用: 把WF方法合成后的er转成信号 

- 13.11 执行 6.2.2.rl_signal_backtest.py  wfm
    - 作用:  WF方法对信号进行绩效评估

- 13.12 执行 6.2.2.rl_signal_backtest.py  wfb
    - 作用: WF方法转成交易规则进行回测


