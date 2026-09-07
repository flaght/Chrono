# Orion 因子挖掘器

`miner` 提供三种公式因子挖掘后端，统一使用 Polars 执行公式，并以截面 IC 或
IC Sharpe 评价候选因子。

| 后端 | 搜索方法 | 主要依赖 | 适用场景 |
|---|---|---|---|
| `bayesian` | Optuna TPE 贝叶斯寻优 | Optuna | 中小搜索空间下的高效结构和参数寻优 |
| `evolution` | 公式树遗传进化 | Polars | 通过交叉和变异探索组合公式 |
| `alphagpt` | Transformer + REINFORCE | PyTorch | 学习公式 Token 的生成策略 |

## 公共架构

```text
miner/
├── formula.py             # Formula AST 和 Polars 公式编译器
├── evaluate.py            # 截面 IC、IC 标准差和 IC Sharpe 评估
├── operators.py           # 公共算子池、配置校验和注册表适配
├── bayesian/              # Optuna 贝叶斯寻优后端
├── evolution/             # 遗传进化后端
└── alphagpt/              # AlphaGPT 策略模型后端
```

三个后端共享：

- `Formula` 公式树和稳定的 `factor_id`；
- `FormulaCompiler` 分层物化编译器；
- `evaluate_formula` 截面 Rank IC 评价；
- `OperatorPool` 算子分类和白名单校验；
- `Launcher(...).optimize(df_lazy, ...)` 调用形式；
- `free` 自由挖掘和 `directed` 定向挖掘模式。

输入 `pl.LazyFrame` 至少需要包含：

- `trade_time`：交易时间；
- `code`：品种标识；
- `feature_names` 中声明的基础特征列；
- `return_column` 指定的未来收益列。

未来收益只能作为挖掘目标，不能进入 `feature_names` 或正式因子公式。

## 挖掘模式

### 自由挖掘

自由挖掘允许公式从 `feature_names` 中任意选择叶子特征：

```python
mode="free"
```

### 定向挖掘

两种模式都把 `feature_names` 作为唯一允许使用的特征集合：

```python
feature_names=("close", "volume", "openint")
```

如果只允许使用持仓量，应直接写成：

```python
feature_names=("openint",)
```

定向模式还要求显式传入 `operator_config`，未填写的算子分类视为空，不会用
默认值补齐。因此定向公式满足：

```text
formula.features ⊆ feature_names
formula.operators ⊆ operator_config
```

完整定向配置示例：

```python
mode="directed"
feature_names=("openint", "close", "volume")
operator_config={
    "window": ("pct_change", "ts_mean", "ts_rank"),
    "binary": ("safe_div",),
    "pair": ("ts_corr",),
}
```

自由模式也遵守 `feature_names`，但 `operator_config` 可以省略，此时使用全部
公共默认算子。自由模式传入部分算子配置时，未填写分类仍使用默认值补齐。

## 公共算子配置

三个后端使用相同的 `operator_config` 格式：

```python
operator_config = {
    "unary": ("abs", "sign"),
    "window": ("pct_change", "ts_mean", "ts_rank"),
    "binary": ("safe_div", "maximum"),
    "pair": ("ts_corr",),
}
```

配置会与 `feature.utils.formula.OPERATORS` 取交集。当前部署没有注册的算子会被
跳过并产生警告，配置中不属于公共白名单的算子会直接报错。

## Bayesian 贝叶斯寻优

Bayesian 使用 Optuna TPE 搜索公式结构、算子和周期。

```python
import polars as pl

from miner.bayesian import BayesianConfig, Launcher

source = pl.scan_ipc("market.feather")
miner = Launcher(
    feature_names=("open", "high", "low", "close", "volume", "openint"),
    return_column="nxt1_ret_5h",
    mode="directed",
    config=BayesianConfig(
        periods=(5, 10, 20, 40, 60),
        max_depth=3,
        n_trials=1000,
        n_jobs=4,
        score="abs_ic_mean",
        log_interval=50,
    ),
    operator_config=operator_config,
)
ranking = miner.optimize(source, top_n=100)
```

旧调用方式中的 `periods`、`max_depth`、`min_observations`、`score` 和
`sampler_seed` 构造参数仍然兼容，新代码建议统一使用 `BayesianConfig`。

## Evolution 遗传进化

Evolution 使用公式树作为遗传个体，支持锦标赛选择、精英保留、子树交叉、
子树变异、点变异和提升变异。

```python
from miner.evolution import EvolutionConfig, Launcher

miner = Launcher(
    feature_names=("open", "high", "low", "close", "volume", "openint"),
    return_column="nxt1_ret_5h",
    mode="directed",
    config=EvolutionConfig(
        population_size=200,
        generations=30,
        tournament_size=5,
        elite_size=10,
        max_depth=3,
        score="abs_ic_mean",
    ),
    operator_config=operator_config,
)
ranking = miner.optimize(source, top_n=100)
```

## AlphaGPT 策略模型

AlphaGPT 使用 actor-critic Transformer 生成后缀公式 Token，通过 REINFORCE
更新策略。PyTorch 只负责模型训练，公式执行和评分仍由 Polars 完成。

```python
from miner.alphagpt import AlphaGPTConfig, Launcher

miner = Launcher(
    feature_names=("open", "high", "low", "close", "volume", "openint"),
    return_column="nxt1_ret_5h",
    mode="directed",
    config=AlphaGPTConfig(
        periods=(2, 4, 6, 8, 10, 12, 20),
        train_steps=100,
        batch_size=64,
        max_formula_len=4,
        score="abs_ic_mean",
        device="cpu",
        log_interval=1,
    ),
    operator_config=operator_config,
)
ranking = miner.optimize(source, top_n=100)
```

安装 PyTorch：

```bash
uv pip install torch
```

## 命令行示例

```bash
# Bayesian
python -m test.bayesian_formula \
  --mode directed \
  --window-operators pct_change ts_mean ts_rank \
  --binary-operators safe_div \
  --pair-operators ts_corr

# Evolution
python -m test.evolution_formula \
  --input market.feather \
  --features open high low close volume openint \
  --return-column nxt1_ret_5h \
  --mode directed \
  --window-operators pct_change ts_mean ts_rank \
  --binary-operators safe_div --pair-operators ts_corr

# AlphaGPT
python -m test.alphagpt_formula \
  --input market.feather \
  --features open high low close volume openint \
  --return-column nxt1_ret_5h \
  --mode directed \
  --window-operators pct_change ts_mean ts_rank \
  --binary-operators safe_div --pair-operators ts_corr \
  --train-steps 100 --batch-size 64 --top-n 100
```

## 结果字段

不同后端的来源字段分别是 `trial`、`generation` 或 `step`，其余主要字段保持
一致：

| 字段 | 含义 |
|---|---|
| `score` | 当前配置选择的优化分数 |
| `formula_id` | 公式结构的稳定短标识 |
| `formula` | 可读公式表达式 |
| `features` | 公式依赖的基础特征 |
| `ic_mean` | 各时点截面 IC 均值 |
| `abs_ic_mean` | IC 均值绝对值 |
| `ic_sharpe` | IC 均值与标准差之比 |
| `observations` | 有效观测总数 |
| `period_count` | 有效截面期数 |
