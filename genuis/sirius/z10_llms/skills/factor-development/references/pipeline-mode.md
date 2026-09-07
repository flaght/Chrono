# Python 脚本模式

`scripts/run_developer.py` 是 `factor-development` 的自动流水线入口。它与 Codex 直接开发模式互斥：只有脚本模式读取外部模型配置。

## 输入

必填参数：

- `--spec`：`factor-idea-researcher` 生成的 `factor-specification.json`。
- `--output-dir`：一个尚不存在的隔离产物目录。

可选参数：

- `--feature-dictionary`：指定本次唯一可用的业务字段；未传入时自动加载 `references/orion_features.json`。`trade_time`、`code` 作为公共键始终允许。
- `--factor-name`：覆盖说明书中的最终因子名，例如 `tf001`；同时控制文件名、元数据名、输出列名和下游 artifact 名称。未指定时沿用说明书的语义名称。批次目录始终由字段依赖决定。
- `--provider openai|ollama`、`--model`：覆盖环境配置。
- `--env-file`：补充进程环境中不存在的配置。
- `--timeout`：单次流式网络操作超时秒数。
- `--max-retries`：生成或校验失败后的最大重试次数。

示例：

```bash
python3 scripts/run_developer.py \
  --spec ../factor-idea-researcher/runs/oi-reversal-003/factor-specification.json \
  --factor-name tf001 \
  --output-dir ./runs/oi-reversal-003
```

需要限制为某个数据集字段时，再增加：

```bash
--feature-dictionary /path/to/features.json
```

指定字典后，若说明书、生成结果的 `required_input_fields` 或代码中的基础字段引用超出字典，生成尝试会失败并进入有限重试；达到最大重试次数后停止。

## 模型配置

脚本先读取进程环境变量，再使用 `--env-file` 补充尚未设置的变量。

| 变量 | 含义 | 默认值 |
| :--- | :--- | :--- |
| `FACTOR_LLM_PROVIDER` | `openai` 或 `ollama` | `openai` |
| `FACTOR_LLM_MODEL` | 模型名称 | OpenAI 为 `gpt-4.1-mini`，Ollama 为 `qwen3:8b` |
| `OPENAI_API_KEY` | OpenAI-compatible API Key | 无，OpenAI 模式必填 |
| `OPENAI_BASE_URL` | OpenAI-compatible API 根地址 | `https://api.openai.com/v1` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `OLLAMA_API_KEY` | Ollama 可选 Bearer Token | 无 |

## 隔离输出

脚本不写正式因子目录。生成并校验成功后的产物位于：

```text
<output-dir>/
├── candidate/feature/<batch>/<factor_name>.py
├── generation-response.raw.txt
├── generation-report.json
├── validation-report.json
├── request.json
└── manifest.json
```

只有候选代码同时通过响应结构、因子静态规则和 Python 编译检查时，生成脚本才以成功状态退出。此时状态是 `awaiting_runtime_validation`，不表示因子已完成全部验证，也不允许进入评估。

代码验证属于 `factor-development` 内部流程，不建立独立代码验证 Skill。随后必须执行受控的最小 LazyFrame 运行测试；人工审批文件是可选增强。运行校验通过后生成 `runtime-validation-report.json` 和 `validated-factor-artifact.json`；只有后者的 `ready_for_evaluation` 为 `true`，下游才可使用。

脚本生成成功不等于因子有效，也不表示可以发布。人工仍需审核公式是否忠实于说明书、字段是否真实可用、窗口和时间对齐是否正确，以及是否存在未来数据泄漏。
