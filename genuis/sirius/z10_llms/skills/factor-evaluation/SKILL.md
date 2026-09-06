---
name: factor-evaluation
description: 从已通过运行验证的因子代码计算因子值并与未来收益隔离对齐，或直接读取已准备的因子值；使用调用参数明确指定的时序或截面方法计算 IC 并应用显式通过规则。不负责研究想法、开发代码或自动选择评估类型。
---

# 因子评估

本 Skill 合并因子值计算、数据对齐、评估函数校验和执行。MVP 固定支持两种评估方法：

- `time_series`：单品种时序评估；
- `cross_section`：多品种截面评估。

它是确定性处理器，不使用提示词或大模型。流程拓扑、输入域隔离和状态语义统一定义在 [references/evaluation-contract.md](references/evaluation-contract.md)。

## 强制选择评估类型

调用方必须通过 `--evaluation-type time_series|cross_section` 明确指定。不得根据 `code` 数量自动选择，不得在一种方法失败后回退到另一种方法。

- `time_series` 要求有效数据只有一个 `code`；
- `cross_section` 要求至少两个 `code`，并在每个 `trade_time` 截面计算 Spearman Rank IC。

数据形态和指定类型不匹配时立即失败。

## 执行方式

本 Skill 是确定性计算 Skill，不需要大模型，不读取 `.env`、`OPENAI_*` 或 `OLLAMA_*`。推荐从 `factor-development` 的审批产物计算因子值：

```bash
python3 scripts/run_evaluation.py \
  --factor-artifact /path/to/validated-factor-artifact.json \
  --market-data /path/to/market-data.parquet \
  --return-data /path/to/future-return.parquet \
  --return-column forward_return \
  --evaluation-type time_series \
  --time-series-evaluator /path/to/evaluate/times/cux001.py \
  --output-dir /path/to/new-output
```

兼容已准备好因子值的入口：

```bash
python3 scripts/run_evaluation.py \
  --input /path/to/factor-data.csv \
  --factor-column my_factor \
  --return-column forward_return \
  --evaluation-type cross_section \
  --output-dir /path/to/new-output
```

完整参数、数据和结果契约见 [references/evaluation-contract.md](references/evaluation-contract.md)。

## 评估规则

1. 代码模式只执行 `validated-factor-artifact.json` 锁定且哈希一致的代码；只把声明的基础字段传入 `compute()`，未来收益在计算结束后才连接。
2. 准备数据模式必须包含 `trade_time`、`code`、因子列和未来收益列。
3. 因子列和未来收益只保留有限数值；缺失和无穷值不参与评估。
4. `time_series` 必须加载调用方指定的 `cux001.py`，兼容 `FactorEvaluatePolars` 和 `FactorEvaluate1`，实际调用 `run()`、`plot_results()` 和 `save_results()`；不会用简化 IC 实现替代完整评估。
5. `cross_section` 在每个时间点对因子和未来收益分别做平均并列排名，计算 Spearman Rank IC，再跨时间汇总。
6. `pass_rule` 可选；提供时必须显式指定指标、比较符和值。不得修改规则以使结果通过。
7. 评估结果只说明统计表现，不代表因果有效、可交易或可以发布。

## 输出

输出目录必须尚不存在。成功后生成：

```text
<output-dir>/
├── request.json
├── evaluation-result.json
├── evaluation-report.json
├── computation-report.json     # 代码模式
├── factor-values.parquet       # 代码模式
├── aligned-evaluation-data.parquet # 代码模式
├── <factor-name>/               # time_series，由 cux001.save_results 生成
│   ├── performance_summary.txt
│   ├── nav.csv
│   ├── ic.csv
│   ├── turnover.csv
│   ├── evaluation_plot.png
│   └── evaluation.xml
├── plot/<factor-name>.png       # time_series
├── xml/<factor-name>.xml        # time_series
├── pass-rule-result.json       # 仅传入 pass rule 时
└── manifest.json
```

返回时报告评估类型、核心指标、样本量、通过规则结果和产物绝对路径。
