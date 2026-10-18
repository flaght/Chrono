# IM期货收益率预测 - 模块化版本

## 概述

本项目实现了完整的IM期货收益率预测流程，采用模块化设计，支持滚动训练方式（Walk-Forward Validation）。

## 模块结构

```
exper1/
├── __init__.py              # 模块初始化文件
├── config.py                # 配置模块（所有参数配置）
├── data_loader.py           # 数据加载模块
├── data_cleaner.py          # 数据清洗模块
├── feature_engineering.py   # 特征工程模块
├── model_trainer.py         # 模型训练模块（支持滚动训练）
├── model_evaluator.py       # 模型评估模块
├── visualizer.py            # 可视化模块
├── main.py                  # 主程序
└── README.md                # 本文件
```

## 核心特性

### 1. 模块化设计
- 每个模块职责单一，易于维护和扩展
- 模块之间通过清晰的接口交互
- 支持独立测试和调试

### 2. 滚动训练（Walk-Forward Validation）
- **这是时间序列预测的正确训练方式**
- 每次用历史数据训练模型，预测未来数据
- 避免数据泄露（不能用未来信息预测过去）
- 模拟真实交易场景，评估模型稳定性

### 3. 详细的中文注释
- 每个模块、类、函数都有详细的中文注释
- 说明参数含义、返回值、使用场景
- 关键步骤都有说明性注释

## 使用方法

### 方式1：直接运行主程序

```python
# 在exper1目录下运行
python main.py
```

### 方式2：作为模块导入

```python
from exper1.main import main
main()
```

### 方式3：使用单个模块

```python
from exper1.data_loader import DataLoader
from exper1.data_cleaner import DataCleaner
from exper1.feature_engineering import FeatureEngineer
from exper1.model_trainer import ModelTrainer
from exper1.model_evaluator import ModelEvaluator
from exper1.visualizer import Visualizer

# 使用各个模块
loader = DataLoader()
df = loader.load('/path/to/your/data.feather')

cleaner = DataCleaner()
df = cleaner.clean(df)

# ... 其他步骤
```

## 配置说明

所有配置参数都在 `config.py` 中，包括：

- **数据配置**：目标变量列名、数据划分比例等
- **数据清洗配置**：NaN阈值、方差阈值等
- **特征工程配置**：相关性阈值、IC阈值等
- **模型训练配置**：LightGBM参数、训练参数等
- **滚动训练配置**：交叉验证折数等
- **评估配置**：预测周期、年化因子等
- **输出配置**：输出目录、图表参数等

## 数据加载

支持多种数据加载方式：

1. **从文件加载**（CSV、Parquet、Feather）
   ```python
   loader = DataLoader()
   df = loader.load('/path/to/data.feather')
   ```

2. **从项目路径加载**（如果项目模块可用）
   ```python
   df = loader.load({
       'method': 'your_method',
       'task_id': 123,
       'instruments': 'IM',
       'period': 15,
       'name': 'final'
   })
   ```

3. **使用模拟数据**（演示用）
   ```python
   loader = DataLoader(use_mock_data=True)
   df = loader.load()
   ```

## 工作流程

1. **数据加载**：从各种数据源加载数据
2. **数据清洗**：处理缺失值、删除无效特征、时间排序
3. **特征工程**：计算IC、智能特征筛选
4. **准备训练数据**：提取特征矩阵和目标变量
5. **滚动训练**：使用Walk-Forward Validation训练模型
6. **模型评估**：计算各种评估指标
7. **特征重要性分析**：分析特征贡献
8. **可视化**：生成评估图表
9. **保存模型**：保存模型和元数据

## 输出文件

运行完成后，会在输出目录（默认：`/mnt/user-data/outputs`）生成以下文件：

- `02_factor_ic.csv` - 因子IC值
- `02_selected_features.csv` - 筛选后的特征列表
- `03_model_evaluation.png` - 模型评估图表（6张子图）
- `03_feature_importance.csv` - 特征重要性
- `03_feature_importance_plot.png` - 特征重要性可视化
- `04_walk_forward_results.csv` - Walk-Forward验证结果
- `05_lgb_model.txt` - 训练好的LightGBM模型
- `05_model_metadata.json` - 模型元数据

## 关键概念

### 滚动训练（Walk-Forward Validation）

滚动训练是时间序列预测的正确方法：

1. **时间序列划分**：按时间顺序划分数据，不能随机划分
2. **逐步向前**：每次用历史数据训练，预测未来数据
3. **避免数据泄露**：不能用未来信息预测过去
4. **模拟真实场景**：更接近实际交易情况

### 评估指标

时序单品预测的核心指标：

1. **方向准确率**：预测涨跌方向是否正确（最关键）
2. **Sharpe Ratio**：策略风险调整后收益
3. **策略累计收益**：模拟交易的总收益
4. **IC/RankIC**：预测值与实际值的相关性
5. **最大回撤**：策略的风险度量

## 注意事项

1. **数据格式要求**：
   - 必须包含 `trade_time`（交易时间）列
   - 必须包含 `code`（合约代码）列
   - 必须包含目标变量列（默认：`nxt1_ret_15h`）

2. **时间序列要求**：
   - 数据必须按时间排序
   - 不能随机划分训练集和测试集
   - 必须使用滚动训练方式

3. **配置修改**：
   - 所有参数都在 `config.py` 中
   - 修改配置后重新运行即可

## 依赖库

- pandas
- numpy
- lightgbm
- scikit-learn
- scipy
- matplotlib
- seaborn

## 版本信息

- 版本：1.0.0
- 作者：Claude
- 日期：2025-01-15

## 许可证

本项目仅供学习和研究使用。

