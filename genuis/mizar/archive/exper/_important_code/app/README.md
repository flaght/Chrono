# 因子绩效展示Web应用

基于Dash框架创建的因子绩效数据展示和探索工具。

## 功能概述

本应用提供了以下核心功能：

1. **因子数据浏览**：从`final.csv`加载因子数据，支持完整的因子列表展示
2. **搜索和筛选**：支持按公式、描述、类别、分数、级别、方向等多维度筛选
3. **绩效图片展示**：自动加载并展示因子的绩效对比图（comparison_plot.png）
4. **绩效摘要**：显示详细的绩效统计信息（performance_summary.txt）
5. **详情查看**：点击表格行可查看因子的完整信息和绩效数据

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据路径

应用默认从以下路径读取数据：
```
records/cicso0/ims/rulex/200037/nxt1_ret_15h/final.csv
```

可以通过以下方式修改路径：

#### 方式1：修改代码中的默认配置

编辑 `app/app.py`，修改 `default_config` 的初始化：

```python
default_config = Config(csv_path='your/path/to/final.csv')
```

#### 方式2：设置环境变量

应用会尝试从环境变量读取路径配置：

- `BASE_PATH`: 基础路径
- `RECORD_PATH`: 记录路径

如果设置了这些环境变量，会自动组合为 `{BASE_PATH}/{RECORD_PATH}`

### 3. 数据文件结构

应用期望的数据结构：

```
records/
└── {method}/
    └── {instruments}/
        └── rulex/
            └── {task_id}/
                └── nxt1_ret_{period}h/
                    ├── final.csv                    # 因子数据文件
                    ├── {source}/                    # category='p' 的因子目录
                    │   └── {factor_id}/
                    │       ├── comparison_plot.png  # 绩效对比图
                    │       └── performance_summary.txt  # 绩效摘要
                    └── d{source}/                   # category='d' 的因子目录
                        └── {factor_id}/
                            ├── comparison_plot.png
                            └── performance_summary.txt
```

### 4. final.csv格式要求

CSV文件必须包含以下列：

- `formula`: 因子公式
- `direction`: 方向（1或-1）
- `source`: 来源标识
- `level`: 级别
- `category`: 类别（'p'、'd'或'f'）
- `score`: 分数
- `detail`: 详细描述（可选）
- `desc`: 描述（可选）

## 运行应用

### 本地开发模式

#### 方式1：使用启动脚本（推荐）

```bash
cd /workspace/worker/pj/Chrono/genuis/mizar
python run_app.py
```

#### 方式2：直接运行模块

```bash
cd /workspace/worker/pj/Chrono/genuis/mizar
python -m app.app
```

#### 方式3：直接运行主文件

```bash
python app/app.py
```

应用会在 `http://0.0.0.0:8050` 启动，可以通过浏览器访问。

#### 环境变量配置

可以通过环境变量配置启动参数：

```bash
export DASH_HOST=0.0.0.0      # 默认值
export DASH_PORT=8050         # 默认值
export DASH_DEBUG=True        # 默认False，设为True启用调试模式
python run_app.py
```

### 生产环境部署

可以使用gunicorn等WSGI服务器部署：

```bash
gunicorn app.app:server --bind 0.0.0.0:8050
```

## 使用说明

### 1. 搜索功能

在搜索框中输入关键词，应用会在以下字段中搜索：
- `formula`: 因子公式
- `detail`: 详细描述
- `desc`: 描述

### 2. 筛选功能

- **Category**: 筛选因子类别
  - `p`: 对比类型（ims vs ics对比图）
  - `d`: 直接类型（单个绩效图）
  - `f`: 特征类型

- **Source**: 按来源筛选（动态加载所有可用来源）

- **Level**: 按级别筛选（1-5）

- **Direction**: 按方向筛选
  - `1`: 向上
  - `-1`: 向下

- **Score Range**: 按分数范围筛选（输入最小值和最大值）

- **Only show factors with images**: 仅显示有绩效图片的因子

### 3. 表格功能

- **排序**: 点击列头可以按该列排序
- **筛选**: 表格支持原生筛选功能
- **分页**: 每页显示20条记录，支持翻页
- **查看详情**: 点击表格中的任意行，会弹出详情模态框

### 4. 详情页面

点击表格行后，会显示：

- **基本信息**：公式、类别、来源、级别、分数、方向、因子ID
- **描述信息**：详细描述和说明
- **绩效图片**：自动加载并显示comparison_plot.png
- **绩效摘要**：显示performance_summary.txt的完整内容

### 5. 统计信息

页面顶部显示三个统计卡片：
- **Total Factors**: 总因子数和当前筛选结果数
- **Avg Score**: 平均分数和分数范围
- **Factors with Images**: 有图片的因子数量和百分比

## 因子处理流程

因子处理流程包括多个阶段，从原始因子挖掘到最终因子选择。以下是各个阶段和对应的处理脚本：

### 流程概述

```
原始因子挖掘 → 双品种对比筛选 → 单品种筛选 → 因子筛选 → 相关性分析 → 最终因子列表
    (eligible)   (复杂因子)      (2.2.3)     (chosen)    (2.2.4)      (final.csv)
                              (简单因子)
```

**筛选策略说明**：
- **逻辑复杂因子**（算子4~6个）：采用双品种对比法，要求在两个品种都满足条件，用于剔除强行过拟合因子
- **逻辑简单因子**（算子1~3个）：采用单品种法，只需在主品种满足条件即可，由 `2.2.3.choose_deform_factors.py` 处理

### 1. 剩余因子筛选和评估 (`2.2.3.choose_deform_factors.py`)

**功能说明**：
- 主要用于筛选剩余因子，即那些在双品种对比中不满足条件，但在主品种中满足条件的简单因子
- 根据时序因子筛选流程，逻辑复杂因子（算子4~6个）采用双品种对比法筛选，而逻辑简单因子（算子1~3个）采用单品种法筛选
- 本脚本筛选的是那些在双品种对比中被过滤掉，但在主品种中满足单品种筛选条件的简单因子
- 对符合条件的因子进行绩效评估，生成绩效对比图
- 将结果保存到 `d{session}` 目录（category='d' 的因子）

**筛选逻辑**：
- **目标因子类型**：表达式简单的因子（算子1~3个之间）
- **筛选条件**：在主品种中满足单品种筛选条件
  - 卡玛大于5
  - IC绝对值大于0.05
  - 年夏普大于1.7
- **排除条件**：已在 `chosen.csv` 中存在的因子（这些因子已经在双品种对比中通过筛选）

**主要步骤**：
1. 从 `{base_path}/{method}/{instruments}/eligible/ic/{task_id}/nxt1_ret_{period}h/{session}/programs_{task_id}_{session}.feather` 加载候选因子
2. 过滤条件：
   - `final_fitness > 0.02`（初始过滤）
   - 不在 `chosen.csv` 中的因子（排除已在双品种对比中通过的因子）
   - `abs(final_fitness) > 0.03`（最终过滤，对应IC绝对值要求）
3. 提取因子依赖的特征字段
4. 加载数据并计算因子值
5. 并行评估因子绩效，生成对比图（在主品种上评估）
6. 保存结果到 `records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/d{session}/{factor_id}/`

**与双品种对比法的关系**：
- 双品种对比法用于筛选逻辑复杂因子（算子4~6个），要求在两个品种（如ims和ics）都满足条件
- 本脚本用于筛选逻辑简单因子（算子1~3个），只需在主品种满足条件即可
- 这样可以充分利用那些在双品种对比中不满足，但在单品种中表现良好的简单因子

**使用方法**：

```bash
python 2.2.3.choose_deform_factors.py
```

**配置参数**（通过 Tactix 配置）：
- `method`: 方法标识（如 'cicso0'）
- `instruments`: 交易品种（如 'ims'）
- `period`: 预测周期（如 15）
- `task_id`: 任务ID（如 200037）
- `session`: 会话标识（如 202521007）

**输入文件**：
- `{base_path}/{method}/{instruments}/eligible/ic/{task_id}/nxt1_ret_{period}h/{session}/programs_{task_id}_{session}.feather`
- `records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/chosen.csv`

**输出目录**：
- `records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/d{session}/{factor_id}/`
  - `comparison_plot.png`: 绩效对比图
  - `performance_summary.txt`: 绩效摘要

### 2. 因子相关性分析和筛选 (`2.2.4.correlation_factors.py`)

**功能说明**：
- 从 `draft.csv` 加载待分析的因子列表
- 计算因子的绩效指标（IC、收益率、Calmar比率等）
- 计算因子之间的相关性（Spearman相关系数）
- 根据相关性阈值过滤高相关因子，保留最优因子
- 输出最终选中的因子列表到 `chosen_{category}_{sort_index}_{threshold}.csv`

**主要步骤**：

#### 2.1 计算绩效指标 (`metrics1` / `metrics2`)
1. 从 `draft.csv` 加载因子列表
2. 计算每个因子的绩效指标：
   - IC均值、IC标准差
   - 平均收益率
   - Calmar比率
   - 其他统计指标
3. 保存绩效数据到 `{base_path}/{method}/{instruments}/correlation/{task_id}/nxt1_ret_{period}h/{instruments}/metrics.csv`
4. 保存因子时间序列数据到 `sequence/{factor_id}.feather`

**区别**：
- `metrics1`: 使用相同的 `instruments` 计算绩效
- `metrics2`: 使用映射的 `instruments` 计算绩效（如 ims → ics）

#### 2.2 相关性分析 (`correlation1` / `correlation2`)
1. 加载绩效指标数据 `metrics.csv`
2. 按排序指标排序（默认：`avg_ret`, `abs_ic`, `calmar`）
3. 加载因子收益率时间序列数据
4. 过滤覆盖率 < 80% 的因子
5. 计算因子间的 Spearman 相关系数
6. 如果相关系数 > 阈值，保留排序靠前的因子，过滤掉排序靠后的因子
7. 验证因子方向一致性（IC方向与direction字段是否一致）
8. 输出最终因子列表

**使用方法**：

```bash
# 计算绩效指标（相同instruments）
python 2.2.4.correlation_factors.py --form metrics1

# 计算绩效指标（映射instruments）
python 2.2.4.correlation_factors.py --form metrics2

# 相关性分析（相同instruments）
python 2.2.4.correlation_factors.py --form correlation1

# 相关性分析（映射instruments）
python 2.2.4.correlation_factors.py --form correlation2
```

**配置参数**（通过 Tactix 配置）：
- `method`: 方法标识
- `instruments`: 交易品种
- `period`: 预测周期
- `task_id`: 任务ID
- `category`: 类别（用于加载收益率数据）
- `sort_index`: 排序索引（默认 "1"）
- `threshold`: 相关性阈值（如 0.7 表示相关系数 > 0.7 时过滤）

**输入文件**：
- `records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/draft.csv`
- `{base_path}/{method}/{instruments}/correlation/{task_id}/nxt1_ret_{period}h/{instruments}/metrics.csv`（相关性分析时）

**输出文件**：
- `{base_path}/{method}/{instruments}/correlation/{task_id}/nxt1_ret_{period}h/{instruments}/metrics.csv`（绩效指标）
- `{base_path}/{method}/{instruments}/correlation/{task_id}/nxt1_ret_{period}h/{instruments}/sequence/{factor_id}.feather`（时间序列数据）
- `records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/chosen_{category}_{sort_index}_{threshold}.csv`（最终因子列表）

**排序指标配置**：
```python
sort_mapping = {
    "1": ["avg_ret", "abs_ic", "calmar"],  # 按平均收益率、绝对IC、Calmar比率排序
}
```

### 3. 数据文件关系

因子处理流程中涉及的主要数据文件：

1. **`draft.csv`**: 待分析的因子列表
   - 位置：`records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/draft.csv`
   - 包含字段：`formula`, `direction`, `source`, `level`, `category`, `score` 等
   - 用途：作为相关性分析的输入

2. **`chosen.csv`**: 已选中的因子列表
   - 位置：`records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/chosen.csv`
   - 包含字段：`formula`, `direction` 等
   - 用途：在因子评估时过滤已评估的因子，避免重复计算

3. **`chosen_{category}_{sort_index}_{threshold}.csv`**: 相关性分析后的最终因子列表
   - 位置：`records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/chosen_{category}_{sort_index}_{threshold}.csv`
   - 包含字段：`id`, `formula`, `direction`
   - 用途：经过相关性过滤后的最终因子选择结果

4. **`final.csv`**: 最终因子数据文件（用于Web应用展示）
   - 位置：`records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/final.csv`
   - 包含字段：`formula`, `direction`, `source`, `level`, `category`, `score`, `detail`, `desc` 等
   - 用途：Web应用的数据源，包含所有需要展示的因子信息

### 4. 典型工作流程

1. **因子挖掘阶段**：通过遗传算法等方法在 `eligible` 目录生成候选因子

2. **因子筛选阶段**：
   - **双品种对比筛选**：对逻辑复杂因子（算子4~6个）进行双品种对比，筛选出在两个品种都满足条件的因子，保存到 `chosen.csv`
   
   - **单品种筛选**（剩余因子筛选）：
   ```bash
   # 筛选在双品种对比中不满足，但在主品种中满足的简单因子
   python 2.2.3.choose_deform_factors.py
   ```
   筛选条件：卡玛>5，IC绝对值>0.05，年夏普>1.7

3. **因子整合阶段**：
   - 将双品种对比筛选结果和单品种筛选结果整合，生成 `draft.csv`

4. **相关性分析阶段**：
   ```bash
   # 计算绩效指标
   python 2.2.4.correlation_factors.py --form metrics1
   
   # 进行相关性分析
   python 2.2.4.correlation_factors.py --form correlation1
   ```

5. **最终整合阶段**：
   - 将相关性分析结果整合到 `final.csv`
   - 启动Web应用查看和探索因子

### 5. 注意事项

- **因子ID一致性**：确保因子ID生成逻辑在整个流程中保持一致
- **数据覆盖**：相关性分析会过滤覆盖率 < 80% 的因子
- **方向验证**：相关性分析会自动验证因子方向（IC方向与direction字段）的一致性
- **并行处理**：两个脚本都支持多进程并行处理，提高计算效率
- **文件路径**：确保所有输入文件路径正确，特别是 `base_path` 的配置

## 技术架构

### 模块结构

```
app/
├── __init__.py          # 包初始化
├── app.py              # Dash主应用
├── config.py           # 配置管理
├── data_loader.py      # 数据加载逻辑
├── detail_view.py     # 详情页面组件
└── utils.py           # 工具函数
```

### 核心组件

1. **Config**: 管理路径配置和数据路径解析
2. **DataLoader**: 加载CSV数据，生成因子ID，构建文件路径
3. **DetailView**: 创建详情模态框，处理图片和摘要显示
4. **Utils**: 因子ID生成（MD5 hash），路径构建，文件检查

### 因子ID生成

应用使用MD5 hash生成因子ID：

```python
from app.utils import create_name_id
factor_id = create_name_id(expression)  # 生成16位十六进制ID
```

ID生成逻辑：
1. 对因子公式进行MD5哈希
2. 取哈希值的前16位作为因子ID

### 路径构建逻辑

- **Category 'p'**: `records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/{source}/{factor_id}/comparison_plot.png`
- **Category 'd'**: `records/{method}/{instruments}/rulex/{task_id}/nxt1_ret_{period}h/d{source}/{factor_id}/comparison_plot.png`
- **Category 'f'**: 使用与'p'相同的路径模式

## 故障排除

### 问题1：数据加载失败

**症状**: 页面显示"No data available"

**解决方案**:
1. 检查CSV文件路径是否正确
2. 确认CSV文件包含所有必需的列
3. 查看控制台错误信息

### 问题2：图片无法显示

**症状**: 详情页面中图片显示为占位符

**解决方案**:
1. 检查图片文件是否存在
2. 确认路径构建是否正确（检查category、source、factor_id）
3. 查看浏览器控制台的错误信息

### 问题3：应用无法启动

**症状**: 启动时出现导入错误

**解决方案**:
1. 确认已安装所有依赖：`pip install -r requirements.txt`
2. 检查Python版本（推荐3.8+）
3. 确认工作目录正确

### 问题4：因子ID不匹配

**症状**: 找不到对应的因子图片或摘要文件

**解决方案**:
1. 确认因子ID生成逻辑与数据生成时使用的逻辑一致
2. 检查因子公式是否完全匹配（包括空格和大小写）

## 扩展和自定义

### 添加新的筛选条件

编辑 `app/data_loader.py` 中的 `filter_factors` 函数，添加新的筛选参数。

### 修改表格显示

编辑 `app/app.py` 中的 `update_table` 回调函数，修改列定义或样式。

### 自定义详情页面

编辑 `app/detail_view.py` 中的 `create_detail_modal` 函数，添加或修改显示内容。

## 许可证

[根据项目需要添加许可证信息]

## 更新日志

### v1.1.0 (2024-12-XX)
- 新增因子处理流程文档
- 添加 `2.2.3.choose_deform_factors.py` 使用说明
- 添加 `2.2.4.correlation_factors.py` 使用说明
- 完善数据文件关系说明
- 添加典型工作流程指南

### v1.0.0 (2024-10-27)
- 初始版本
- 支持因子数据加载和展示
- 支持搜索、筛选、排序功能
- 支持详情页面和绩效图片展示

