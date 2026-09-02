### 数据标识说明
1. b 开头为通联现成bar数据
2. r 开头为自行聚合bar数据
3. t 开头为实盘聚合bar数据

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

- 4.  执行 2.2.3.choose_deform_factors.py all
    - 作用 根据上述筛选出来的因子 在训练集和校验集 生成绩效图

- 5.  21.2.人工初筛绩效因子.ipynb
    - 作用: 人工筛选因子

- 6. 执行 2.2.3.choose_deform_factors.py  recent
    - 作用: 根据上述筛选出来的因子 在近期集 生成绩效图，这里会把之前选择中的因子都跑出来，这样的目的是为了末尾淘汰。比如之前标准是夏普大于1.2 后面发现有更高的 改成1.5 这样之前的一些不满足就可以删掉

- 7. 21.3.人工再筛绩效因子.ipynb
    - 作用: 通过绩效图筛选近期也表现不错的因子


注意这些选中的因子要进行绩效跟踪

- 8.  执行 3.0.1.preprocess_data.py  build
    - 作用: 筛选的因子进行方向调整和时序标准化

- 9.  执行 3.0.1.preprocess_data.py  prepare
    - 作用: 切割因子数据，创建训练集 校验集 测试集

- 10.1  执行  3.0.2.blend_factors.py  selected
    - 作用: 根据剔除高相关性因子

- 10.2 执行  3.0.2.blend_factors.py  composite
    - 作用: 选中因子的等权合成绩效情况


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














## 1. 因子构建 (Factor Construction)

### 1.1 基础因子创建

**文件位置**: `1.1.1.create_basic_factors.py`

**功能说明**:
- 从Lumina库中创建基础因子（i001-i014系列）
- 使用各种技术指标函数计算因子值
- 将因子数据保存为Feather格式

**核心流程**:
```python
1. 加载市场数据（fetch_main_market）
2. 遍历因子模块（i001-i01xx）
3. 计算每个因子（calc_impulse）
4. 保存因子数据到 records/{method}/{instruments}/factors/{name}_factors.feather
```

**数据存储结构**:
```
records/
  {method}/              # 方法名称（如 cicso0, bicso0）
    {instruments}/        # 品种代码（如 ims, rbb）
      factors/            # 因子数据目录
        i001_factors.feather
        i002_factors.feather
        ...
```

**关键函数**:
- `callback_save()`: 保存因子数据到指定目录
- `calculate_factors()`: 遍历并计算所有因子模块
- `merge()`: 合并所有因子数据，生成统一的factors_data.feather


### 1.2 不定向挖掘因子
**文件位置**: `2.1.2.gentic_motor_factors.py`

**功能说明**:
- 使用GA算法自动生成和优化因子表达式
- 通过进化过程筛选表现良好的因子
- 支持自定义参数和适应度函数

**核心流程**:
```python
1. 初始化遗传算法引擎
2. 生成候选因子（通过遗传操作：交叉、变异等）
3. 计算适应度（fitness）
4. 筛选优秀因子（final_fitness > threshold）
5. 保存因子程序到 programs_{task_id}_{session}.feather
```

### 1.3 定向进化因子挖掘

**文件位置**: `2.1.3.directed_motor_factors.py`

**功能说明**:
- 基于已有优秀因子进行定向进化
- 通过相关性过滤和增益筛选优化因子集合
- 支持sequential_gain算法进行因子去重

**核心特性**:
- 相关性过滤：剔除高相关性因子
- IC阈值过滤：保留IC值超过阈值的因子
- 增益阈值：基于增益指标筛选因子

**关键函数**:
- `callback_models()`: 处理每代进化结果
- `sequential_gain()`: 基于增益的因子筛选

### 1.4 定向优化因子挖掘

**文件位置**: 
- `2.1.4.optuna_parellel_factors.py` (并行优化)
- `2.1.5.optuna_motor_factors.py` (单进程优化)

**功能说明**:
- 基于已有因子表达式进行定向进化
- 使用Optuna框架进行超参数优化
- 支持贝叶斯优化和网格搜索
- 自动寻找最优参数组合

**依赖资源**:
- `records/resource/expression_dependencies.csv`: 表达式依赖关系
- `records/resource/level2_fields_dependencies.csv`: 字段依赖关系

---

### 2.1 筛选流程概述

因子筛选采用多阶段筛选策略，逐步提高筛选标准：

```
挖掘算法因子 (gentic)
    ↓ [筛选标准: calmar>5, sharpe2>1.5, abs_ic>0.02]
合格因子 (eligible)
    ↓ [筛选标准: calmar>3, sharpe2>1.0, abs_ic>0.02]
验证因子 (valid)
    ↓ [最终评估]
最终因子 (final.csv)
```


### 2.2 筛选脚本

**文件位置**: `2.2.1.scope_valid_factors.py`

**功能说明**:
- 从gentic目录加载候选因子
- 基于绩效指标进行筛选
- 保存筛选后的因子到eligible或valid目录

**筛选指标**:
- **Calmar Ratio**: 年化收益/最大回撤
- **Sharpe2**: 调整后的夏普比率
- **Abs IC**: 绝对信息系数

**第一阶段筛选** (`run1`):
```python
筛选标准:
- calmar > 5
- sharpe2 > 1.5
- abs_ic > 0.02

输出目录: records/{method}/{instruments}/eligible/ic/
```

**第二阶段筛选** (`run2`):
```python
筛选标准:
- calmar > 3
- sharpe2 > 1.0
- abs_ic > 0.02
- 跨品种验证（如ims -> ics）

输出目录: records/{method}/{instruments}/valid/ic/
```

**核心函数**:
- `load_factors()`: 加载因子程序数据
- `valid_programs()`: 验证因子绩效指标
- `run()`: 执行筛选流程


---

## 3. 因子绩效存储 (Factor Performance Storage)

### 3.1 存储目录结构

**标准路径格式**:
```
records/
  {method}/              # 方法名称（如 cicso0）
    {instruments}/       # 品种代码（如 ims）
      rulex/             # 因子规则目录
        {task_id}/       # 任务ID
          nxt1_ret_{period}h/  # 预测周期
            {source}/    # 因子来源ID（如 202510225）
              {factor_id}/     # 因子ID（MD5哈希）
                comparison_plot.png      # 绩效对比图（category='p'）
                evaluation_plot.png      # 评估图（category='d'）
                performance_summary.txt  # 绩效摘要
```

**特殊目录结构**:
- **Category 'p'**: `{source}/{factor_id}/`
- **Category 'd'**: `d{source}/{factor_id}/`

### 3.2 最终因子清单文件

**文件位置**: `records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/final.csv`

**文件格式** (CSV):
```csv
formula,direction,source,level,category,score,detail,desc
"MA(60,MCPS(90,'smart_tick_in'))",-1,202510225,1,p,8,捕捉聪明资金买入行为...,对聪明钱买入笔数特征...
"DELTA(60,MIR(60,'smart_tick_in_pct'))",1,202510225,1,p,8,捕捉聪明资金占比...,对聪明钱买入笔数占比...
```

**字段说明**:
- `formula`: 因子表达式
- `direction`: 因子方向（-1或1）
- `source`: 因子来源ID（session ID）
- `level`: 因子层级
- `category`: 因子类别（'p'或'd'）
- `score`: 因子评分
- `detail`: 因子详细描述
- `desc`: 因子说明

### 3.3 绩效摘要文件

**文件位置**: `{factor_id}/performance_summary.txt`

**文件格式**:
```
Expression: EMA(10,EMA(10,MT3(90,EMA(10,MCPS(5,MT3(90,MCPS(5,'open')))))))
Name: 10765938

--- Performance Metrics ---
Avg Return (bps)         : 0.51
Total Return             : 58.17%
Sharpe Ratio             : 0.04
Ann Sharpe Ratio         : 1.68
Max Drawdown             : -6.81%
Calmar Ratio             : 8.55
Win Rate                 : 50.63%
Profit/Loss Ratio        : 1.16

--- Factor Characteristics ---
IC Mean                  : -0.0369
ICIR                     : -0.1422
Mean Turnover            : 0.1463
Factor Autocorr          : 0.8691
Return Autocorr          : -0.0502
```

### 3.4 绩效图像文件

**图像类型**:
- `comparison_plot.png`: 用于category='p'的因子，显示因子值与收益的对比
- `evaluation_plot.png`: 用于category='d'的因子，显示评估结果

**生成位置**:
- 在因子评估过程中自动生成
- 保存在每个因子ID对应的子目录中

### 3.5 绩效数据存储

**存储位置**: `records/{method}/{instruments}/returns/`

**数据格式**: Feather格式

**数据结构**:
- 按数据集分类：train, val, test
- 包含时间序列收益数据

---