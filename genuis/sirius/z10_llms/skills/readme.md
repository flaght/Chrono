# KD-HYS 四个 Skill 双模式调用使用手册

## 1. 文档目的

本文说明以下四个 Skill 如何使用 Codex 直接模式和 Python 脚本模式：

1. `factor-idea-researcher`：把因子想法转化为完整因子说明书；
2. `factor-development`：把因子说明书转化为候选因子代码，并通过测试数据完成运行验证；
3. `factor-evaluation`：计算或读取因子值，执行时序或截面评估；
4. `factor-mining-orchestrator`：串联前三个阶段，管理状态、暂停点和恢复。

这里的“双模式”需要区分两种情况：

| Skill 类型 | Codex 直接模式 | Python 脚本模式 |
| :--- | :--- | :--- |
| 需要大模型的开发型 Skill | 当前 Codex 直接研究或开发，不读取 `.env` | 脚本调用 `.env` 中配置的 OpenAI-compatible 或 Ollama |
| 确定性 Skill | Codex 按 Skill 调用确定性程序并解释结果 | 用户直接运行同一个确定性 Python 程序 |

因此，`factor-evaluation` 和 `factor-mining-orchestrator` 自身不需要额外大模型。总控可以调用需要模型的子 Skill，但不会自己生成研究结论或代码。

## 2. 目录和通用约定

本文假设当前目录为：

```bash
cd /Users/kerry/work/orion/demo/ht-hys/kd-hys
```

若 Skill 已复制到其他项目，例如：

```text
/workspace/project/skills/
```

请把命令中的相对路径替换为实际路径。推荐把不同阶段的运行产物统一放在项目的 `warehouse/runs/` 下：

```text
warehouse/runs/<experiment-name>/
```

所有新运行目录默认必须不存在。不要用相同目录覆盖上一次实验；需要恢复的只有总控运行目录。

### 2.1 外部模型配置

只有下面两个脚本读取模型配置：

- `factor-idea-researcher/scripts/run_researcher.py`；
- `factor-development/scripts/run_developer.py`。

OpenAI-compatible 示例：

```dotenv
FACTOR_LLM_PROVIDER=openai
FACTOR_LLM_MODEL=你的模型名称

OPENAI_API_KEY=你的APIKey
OPENAI_BASE_URL=https://你的服务地址/v1
```

Ollama 示例：

```dotenv
FACTOR_LLM_PROVIDER=ollama
FACTOR_LLM_MODEL=qwen3:8b

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=
```

安全要求：

- `.env` 不得提交到代码仓库；
- API Key 不得写入总控请求 JSON；
- Codex 直接模式不得读取或使用 `.env`；
- `--max-retries 2` 表示首次请求之外最多再尝试两次，总请求次数最多三次。

## 3. `factor-idea-researcher`

### 3.1 输入和主产物

输入：一个因子想法、研究来源模式，以及显式指定或由 Skill 内置兜底的特征字典。

主产物：

```text
factor-specification.json
```

完整运行目录通常包含：

```text
first-principles.json
first-principles.raw.txt
factor-specification.json
factor-specification.raw.txt
research-context.json
validation-report.json
request.json
manifest.json
```

### 3.2 Codex 直接模式

在 Codex 中明确指定 Skill 和输出目录，例如：

```text
请使用 factor-idea-researcher，以 Codex 直接模式研究下面的因子想法：
“持仓量快速增长但价格动量衰减可能预示反转”。

研究来源使用 direct，不读取 .env，不调用 scripts/run_researcher.py。
请把产物写入：
/path/to/warehouse/runs/test-idea-codex-001
```

使用检索模式：

```text
请使用 factor-idea-researcher，以 Codex 直接模式和 search 来源研究这个想法。
使用 Codex 当前可用的网页检索能力，不调用 Tavily/Jina 脚本适配器。
输出到 /path/to/warehouse/runs/test-idea-search-001。
```

Codex 应读取该 Skill 的 Prompt 契约，生成文件后执行：

```bash
python3 factor-idea-researcher/scripts/validate_research_artifacts.py \
  --first-principles /path/to/warehouse/runs/test-idea-codex-001/first-principles.json \
  --factor-specification /path/to/warehouse/runs/test-idea-codex-001/factor-specification.json
```

如果该运行使用了特征字典，还要向校验命令传入同一个 `--feature-dictionary`。如果校验脚本的实际帮助信息发生变化，以 `--help` 为准。

### 3.3 Python 脚本模式

直接研究，不使用检索和抓取：

```bash
python3 factor-idea-researcher/scripts/run_researcher.py \
  --idea "持仓量快速增长但价格动量衰减可能预示反转" \
  --mode direct \
  --env-file ./factor-idea-researcher/.env \
  --timeout 1200 \
  --max-retries 2 \
  --output-dir ./warehouse/runs/test-idea-001
```

公开资料检索：

```bash
python3 factor-idea-researcher/scripts/run_researcher.py \
  --idea "持仓量快速增长但价格动量衰减可能预示反转" \
  --mode search \
  --query "open interest growth momentum decay reversal factor" \
  --max-results 5 \
  --env-file ./factor-idea-researcher/.env \
  --timeout 1200 \
  --max-retries 2 \
  --output-dir ./warehouse/runs/test-idea-search-001
```

抓取指定网页：

```bash
python3 factor-idea-researcher/scripts/run_researcher.py \
  --idea "从指定研究资料构建持仓量背离因子" \
  --mode fetch \
  --url "https://example.com/research-a" \
  --url "https://example.com/research-b" \
  --env-file ./factor-idea-researcher/.env \
  --timeout 1200 \
  --max-retries 2 \
  --output-dir ./warehouse/runs/test-idea-fetch-001
```

传入特征字典：

```bash
python3 factor-idea-researcher/scripts/run_researcher.py \
  --idea "持仓量快速增长但价格动量衰减可能预示反转" \
  --mode direct \
  --feature-dictionary /path/to/features.json \
  --env-file ./factor-idea-researcher/.env \
  --output-dir ./warehouse/runs/test-idea-feature-001
```

字段字典采用“显式文件优先，Skill 内置兜底”：

- 传入 `--feature-dictionary /path/to/features.json`：加载指定字段文件；
- 不传：自动加载 `factor-idea-researcher/references/orion_features.json`；
- 两种情况都属于约束模式，说明书只能使用实际字典中的字段，缺失能力进入 `missing_features`；
- 每次研究都会把实际使用的字典保存为输出目录中的 `feature-dictionary.json`。

### 3.4 检查结果

```bash
python3 -m json.tool \
  ./warehouse/runs/test-idea-001/factor-specification.json

python3 -m json.tool \
  ./warehouse/runs/test-idea-001/validation-report.json
```

确认：

- 终端最终输出 `"success": true`；
- `factor_name` 是英文 snake_case；
- `required_features` 不为空；
- `missing_features` 符合真实数据能力；
- 没有声称已经完成回测或取得 IC。

## 4. `factor-development`

### 4.1 输入和审核边界

输入：上一步的 `factor-specification.json`。

Python 模式第一次运行只生成候选代码并完成静态、契约和编译检查。成功状态是：

```text
awaiting_runtime_validation
```

这不代表代码可以进入评估。还需要使用明确提供的测试数据执行受控运行验证；人工审批是可选增强。

### 4.2 Codex 直接模式

在 Codex 中调用：

```text
请使用 factor-development，以 Codex 直接模式，根据下面的说明书开发因子：
/path/to/warehouse/runs/test-idea-001/factor-specification.json

不要读取 .env，不要调用 scripts/run_developer.py。
先生成隔离候选代码并执行静态和编译检查，然后使用我明确提供的测试数据进行受控运行验证。
输出目录：/path/to/warehouse/runs/test-development-codex-001
```

准备好测试数据后继续告诉 Codex：

```text
请继续使用 factor-development，
用 /path/to/test-market-data.parquet 执行最小 LazyFrame 运行验证，
并生成 validated-factor-artifact.json。
```

Codex 直接模式使用当前 Codex 完成代码开发，不调用环境模型。运行验证成功即可生成下游凭证。

### 4.3 Python 脚本模式：生成候选代码

```bash
python3 factor-development/scripts/run_developer.py \
  --spec ./warehouse/runs/test-idea-001/factor-specification.json \
  --factor-name tf001 \
  --env-file ./factor-development/.env \
  --timeout 1200 \
  --max-retries 2 \
  --output-dir ./warehouse/runs/test-development-001
```

如果需要覆盖研究产物携带的字段字典，开发阶段可以显式传入：

```bash
  --feature-dictionary /path/to/features.json
```

未传入时，`factor-development` 自动加载自身的
`factor-development/references/orion_features.json`；通过总控运行时，总控会把研究阶段实际输出的 `feature-dictionary.json` 自动传给开发阶段，保证两个阶段字段一致。

开发阶段的字段限制同时作用于三层：

1. 校验说明书的 `required_features`；
2. 传给模型的 `allowed_input_fields`；
3. 校验生成结果的 `required_input_fields` 和代码中的基础字段引用。

因此指定字段文件后，生成代码不能使用该文件之外的业务字段；公共键
`trade_time` 和 `code` 始终允许。

候选代码位于：

```text
warehouse/runs/test-development-001/candidate/feature/<batch>/<factor_name>.py
```

检查：

```bash
python3 -m json.tool \
  ./warehouse/runs/test-development-001/manifest.json

python3 -m json.tool \
  ./warehouse/runs/test-development-001/validation-report.json
```

必须看到：

```json
{
  "status": "awaiting_runtime_validation",
  "ready_for_evaluation": false
}
```

### 4.4 可选人工审批文件

从开发运行的 `manifest.json` 读取 `factor_sha256`，创建审批文件，例如：

```json
{
  "approved": true,
  "reviewer": "kerry",
  "approved_at": "2026-09-01T10:00:00+08:00",
  "factor_sha256": "复制 manifest.json 中的 factor_sha256"
}
```

保存为：

```text
warehouse/runs/test-development-001/approval.json
```

需要人工审计时可以创建审批文件。建议检查：

- 公式与说明书一致；
- 输入字段真实存在；
- 没有未来函数；
- 所有时序运算按 `code` 分组；
- 没有数据加载、网络请求、`.collect()` 或发布操作；
- 输出只有 `trade_time`、`code` 和因子列。

### 4.5 Python 脚本模式：运行验证

```bash
python3 factor-development/scripts/finalize_validation.py \
  --run-dir ./warehouse/runs/test-development-001 \
  --test-data /path/to/test-market-data.parquet
```

这是最小运行验证方式。执行成功时生成 `validation_mode: runtime_only`。若需要人工审批审计，再增加：

```bash
--approval ./warehouse/runs/test-development-001/approval.json
```

成功后生成：

```text
runtime-validation-report.json
validated-factor-artifact.json
```

检查：

```bash
python3 -m json.tool \
  ./warehouse/runs/test-development-001/validated-factor-artifact.json
```

必须满足：

```json
{
  "status": "validated",
  "ready_for_evaluation": true
}
```

## 5. `factor-evaluation`

### 5.1 双入口含义

该 Skill 不需要大模型：

- Codex 直接模式：让 Codex检查参数、运行确定性评估脚本并解释结果；
- Python 脚本模式：用户直接运行 `run_evaluation.py`。

两种方式使用同一套评估代码，因此不会出现 Codex 计算和脚本计算结果不一致的问题。

### 5.2 Codex 直接模式

```text
请使用 factor-evaluation 评估下面的已验证因子：
/path/to/validated-factor-artifact.json

行情数据：/path/to/market-data.parquet
未来收益：/path/to/future-return.parquet
收益列：forward_return
评估类型：time_series
输出目录：/path/to/warehouse/runs/test-evaluation-codex-001

请运行确定性脚本，不读取 .env，不自动切换评估类型。
```

### 5.3 Python 代码模式：从因子代码计算因子值

时序评估：

```bash
python3 factor-evaluation/scripts/run_evaluation.py \
  --factor-artifact ./warehouse/runs/test-development-001/validated-factor-artifact.json \
  --market-data /path/to/market-data.parquet \
  --return-data /path/to/future-return.parquet \
  --return-column forward_return \
  --evaluation-type time_series \
  --time-series-evaluator /path/to/evaluate/times/cux001.py \
  --min-observations 100 \
  --roll-win 252 \
  --min-periods 5 \
  --output-dir ./warehouse/runs/test-evaluation-ts-001
```

截面评估：

```bash
python3 factor-evaluation/scripts/run_evaluation.py \
  --factor-artifact ./warehouse/runs/test-development-001/validated-factor-artifact.json \
  --market-data /path/to/multi-code-market-data.parquet \
  --return-data /path/to/multi-code-future-return.parquet \
  --return-column forward_return \
  --evaluation-type cross_section \
  --min-observations 100 \
  --min-cross-section-size 2 \
  --output-dir ./warehouse/runs/test-evaluation-cs-001
```

代码模式下：

1. 校验审批产物和因子代码 SHA256；
2. 只把 `required_input_fields` 传给 `compute()`；
3. 因子值计算完成后才连接未来收益；
4. 保存因子值和对齐数据。

### 5.4 Python 准备数据模式

如果已经有包含因子值与未来收益的数据：

```bash
python3 factor-evaluation/scripts/run_evaluation.py \
  --input /path/to/prepared-factor-data.csv \
  --factor-column oi_growth_momentum_decay_reversal \
  --return-column forward_return \
  --evaluation-type cross_section \
  --min-observations 100 \
  --output-dir ./warehouse/runs/test-evaluation-prepared-001
```

准备数据必须包含：

```text
trade_time
code
<factor-column>
<return-column>
```

### 5.5 Pass Rule

示例 `pass-rule.json`：

```json
{
  "metric": "abs_ic_mean",
  "operator": ">=",
  "value": 0.03
}
```

调用时增加：

```bash
--pass-rule /path/to/pass-rule.json
```

不得为了让结果通过而自动修改阈值。

### 5.6 输出检查

代码模式会生成：

```text
factor-values.parquet
aligned-evaluation-data.parquet
computation-report.json
evaluation-result.json
evaluation-report.json
manifest.json
```

查看结果：

```bash
python3 -m json.tool \
  ./warehouse/runs/test-evaluation-ts-001/evaluation-result.json
```

注意：

- `time_series` 只接受一个 `code`；
- `cross_section` 至少需要两个 `code`；
- 类型必须显式指定，不会自动判断或失败后回退。

## 6. `factor-mining-orchestrator`

### 6.1 双模式含义

总控本身是确定性状态机：

- Codex 直接模式：当前 Codex逐阶段使用三个子 Skill，维护文件化状态；
- Python 脚本模式：`run_orchestrator.py` 调用子脚本并维护 `pipeline-manifest.json`。

请求中的研究和开发阶段分别使用：

```json
"execution": "codex"
```

或：

```json
"execution": "python"
```

Python 总控无法直接调用当前 Codex。遇到 `execution: codex` 且没有阶段产物时，会正常暂停：

```text
awaiting_codex_research
awaiting_codex_development
```

### 6.2 Codex 直接模式

```text
请使用 factor-mining-orchestrator，以 Codex 直接模式运行一次因子挖掘流程。

因子想法：持仓量快速增长但价格动量衰减可能预示反转
研究来源：direct
评估类型：time_series
行情数据：/path/to/market-data.parquet
未来收益：/path/to/future-return.parquet
收益列：forward_return
运行目录：/path/to/warehouse/runs/pipeline-codex-001

研究和开发阶段使用当前 Codex，不读取 .env；候选代码生成后使用指定测试数据执行运行验证。
```

Codex 应分别遵守三个子 Skill 的契约，而不是在总控中复制研究 Prompt、代码规则或评估算法。

### 6.3 Python 一键模式：只输入想法

部署时只需编辑一次：

```text
factor-mining-orchestrator/config/pipeline.defaults.json
```

建议配置为：

```json
{
  "schema_version": "1.0.0",
  "research": {
    "execution": "python",
    "mode": "direct",
    "timeout": 1200,
    "max_retries": 2
  },
  "development": {
    "execution": "python",
    "factor_name": "tf001",
    "timeout": 1200,
    "max_retries": 2,
    "test_data": "/path/to/test-market-data.parquet"
  },
  "evaluation": {
    "evaluation_type": "time_series",
    "market_data": "/path/to/market-data.parquet",
    "return_data": "/path/to/future-return.parquet",
    "return_column": "forward_return",
    "time_series_evaluator": "/path/to/evaluate/cux001.py",
    "min_observations": 100,
    "roll_win": 15,
    "min_periods": 5,
    "resampling_win": 5,
    "fee": 0.0,
    "scale_method": "roll_min_max",
    "annualization_factor": 252
  }
}
```

以后每次只输入想法和一个全新的运行目录：

```bash
python3 factor-mining-orchestrator/scripts/run_orchestrator.py \
  --idea "持仓量快速增长但价格动量衰减可能预示反转" \
  --run-dir ./warehouse/runs/pipeline-test-001
```

想法较长时可以使用 UTF-8 的 `.txt`、`.md` 或 `.markdown` 文件：

```bash
python3 factor-mining-orchestrator/scripts/run_orchestrator.py \
  --input ./ideas/oi-reversal.md \
  --factor-name tf001 \
  --run-dir ./warehouse/runs/pipeline-test-002
```

输入文件也可以放在运行目录中。目录里只有该文本等普通文件时，总控会自动创建
`pipeline-manifest.json`；无需提前创建 manifest。

预期流程：

```text
researching
  ↓
developing
  ↓
validating
  ↓
evaluating
  ↓
completed
```

研究脚本和开发脚本默认读取各自 Skill 根目录的 `.env`，因此无需在总控配置中重复填写 `env_file`。总控生成候选代码后自动调用 `finalize_validation.py`；只要测试数据验证成功，就立即进入绩效评估。

检查：

```bash
python3 -m json.tool \
  ./warehouse/runs/pipeline-test-001/pipeline-manifest.json
```

### 6.4 单次覆盖与恢复

只有某次实验需要覆盖默认参数时，才创建精简请求 JSON；请求字段会覆盖默认配置。例如只修改评估窗口：

```json
{
  "research": {
    "idea": "持仓量快速增长但价格动量衰减可能预示反转"
  },
  "evaluation": {
    "roll_win": 30,
    "resampling_win": 15
  }
}
```

运行或恢复：

```bash
python3 factor-mining-orchestrator/scripts/run_orchestrator.py \
  --request ./warehouse/requests/pipeline-test-001.json \
  --run-dir ./warehouse/runs/pipeline-test-001
```

若默认配置没有填写 `development.test_data`，总控才会暂停为 `awaiting_runtime_validation`。补齐配置后使用相同命令和运行目录恢复即可；人工审批文件始终可选。

### 6.5 混合模式

研究由 Codex 完成、开发由 Python 完成：

```json
{
  "research": {
    "execution": "codex",
    "specification": "/path/to/codex/factor-specification.json"
  },
  "development": {
    "execution": "python",
    "env_file": "/path/to/factor-development/.env"
  }
}
```

研究由 Python 完成、开发由 Codex 完成：

```json
{
  "research": {
    "execution": "python",
    "idea": "因子想法",
    "mode": "direct",
    "env_file": "/path/to/factor-idea-researcher/.env"
  },
  "development": {
    "execution": "codex"
  }
}
```

第二种配置会在研究完成后进入 `awaiting_codex_development`。Codex 完成开发与验证后，把产物写入：

```json
"validated_factor_artifact": "/path/to/validated-factor-artifact.json"
```

再恢复同一个总控运行。

### 6.6 总控状态含义

| 状态 | 含义 | 下一步 |
| :--- | :--- | :--- |
| `awaiting_codex_research` | 请求指定 Codex 研究，但没有说明书 | 使用 `factor-idea-researcher` 生成说明书 |
| `awaiting_codex_development` | 请求指定 Codex 开发，但没有验证产物 | 使用 `factor-development` 开发并验证 |
| `awaiting_runtime_validation` | 候选代码已生成但没有测试数据 | 提供测试数据；审批文件可选 |
| `failed` | 输入或子进程失败 | 查看 `next_action`，修正请求后恢复 |
| `completed` | 研究、开发验证和评估完成 | 人工检查结果；发布需要独立授权 |

## 7. 推荐的逐步测试顺序

不要第一次就从总控开始排查所有问题。推荐顺序：

```text
1. 单独测试 factor-idea-researcher Python 模式
2. 使用说明书测试 factor-development Python 模式
3. 使用测试数据运行 finalize_validation.py；人工审批可选
4. 使用 validated-factor-artifact.json 测试 factor-evaluation
5. 最后测试 factor-mining-orchestrator 的暂停和恢复
6. 再分别测试 Codex 直接模式和混合模式
```

每一步只在主产物状态正确后进入下一步：

```text
factor-specification.json
  ↓
development manifest: awaiting_runtime_validation
  ↓ 测试数据运行验证
validated-factor-artifact.json: ready_for_evaluation=true
  ↓
evaluation-result.json
  ↓
pipeline-manifest.json: completed
```

## 8. 常见错误

### 输出目录已经存在

研究、开发和单独评估运行要求输出目录不存在。请换一个新编号，例如从 `test-001` 改为 `test-002`。不要为了测试方便覆盖历史产物。

### `OPENAI_API_KEY is required`

只影响研究或开发的 Python 外部模型模式。检查：

- 是否传入正确的 `--env-file`；
- `.env` 是否位于当前执行环境；
- Key 名是否严格为 `OPENAI_API_KEY`；
- `FACTOR_LLM_PROVIDER` 是否与配置一致。

### `feature dictionary not found`

`--feature-dictionary` 后面的路径不存在。需要自定义字段时使用真实绝对路径或有效相对路径；不需要自定义时删除该参数，研究 Skill 会自动使用内置 `references/orion_features.json`。

### 流式生成结束后重试

先确认使用的是当前版本脚本。若仍出现超时：

- 增大 `--timeout`；
- 检查兼容服务是否正确发送流结束事件；
- 查看脚本打印的阶段和请求次数；
- `--max-retries 2` 最多只会执行三次，不应无限循环。

### 评估类型不匹配

- 单一 `code` 使用 `time_series`；
- 多个 `code` 的逐时点比较使用 `cross_section`；
- 系统不会自动切换类型。

### 总控停在运行验证

这是正常状态。在请求中补充 `test_data`，然后用同一个 `--run-dir` 恢复；`approval` 可选。

## 9. 当前范围

当前四个 Skill 已覆盖：

```text
因子想法 → 完整说明书 → 候选代码 → 运行验证
→ 因子值计算 → 时序/截面评估 → 可恢复总控状态
```

当前尚不包括：

- 自动维护正式特征字典；
- 自动发布因子代码；
- 未经人工授权修改正式注册表；
- 自动选择评估类型或自动降低通过阈值。
