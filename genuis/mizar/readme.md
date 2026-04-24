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


3.0.1.blend_factors.py
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