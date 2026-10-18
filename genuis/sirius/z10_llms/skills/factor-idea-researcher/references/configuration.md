# Python 脚本模式配置

## 想法输入

短文本使用 `--idea`；长文本使用 `--input` 指向 UTF-8 的 `.txt`、`.md` 或
`.markdown` 文件。两个参数互斥，正文最多 20,000 字符。例如：

```bash
python3 scripts/run_researcher.py \
  --input /path/to/factor-idea.md \
  --mode direct \
  --output-dir /path/to/new-run
```

未传 `--env-file` 时自动读取 Skill 根目录的 `.env`。

本文件仅适用于 `python3 scripts/run_researcher.py`。Codex 直接模式不得读取或使用这里的环境变量。

脚本优先读取进程环境变量，再使用 `--env-file` 补充尚未设置的配置。

## 通用配置

| 变量 | 含义 | 默认值 |
| :--- | :--- | :--- |
| `FACTOR_LLM_PROVIDER` | `openai` 或 `ollama` | `openai` |
| `FACTOR_LLM_MODEL` | 模型名称 | 随 Provider 选择 |

## OpenAI-compatible 配置

| 变量 | 含义 | 默认值 |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | API Key | 无 |
| `OPENAI_BASE_URL` | OpenAI-compatible API 根地址 | `https://api.openai.com/v1` |

该 Provider 调用 `POST {OPENAI_BASE_URL}/chat/completions`。

## Ollama 配置

| 变量 | 含义 | 默认值 |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | Ollama 服务根地址 | `http://localhost:11434` |
| `OLLAMA_API_KEY` | 受保护服务可选的 Bearer Token | 无 |

该 Provider 调用 `POST {OLLAMA_BASE_URL}/api/chat`。

## 检索与抓取配置

| 变量 | 含义 | 默认值 |
| :--- | :--- | :--- |
| `TAVILY_API_KEY` | `search` 模式必填 | 无 |
| `TAVILY_BASE_URL` | Tavily API 根地址 | `https://api.tavily.com` |
| `JINA_READER_BASE_URL` | `fetch` 模式使用的 Reader 前缀 | `https://r.jina.ai/` |

`fetch` 优先使用 Jina Reader，失败后降级为直接 HTTP GET。只允许抓取公网 HTTP(S) 地址。

## 流式响应与超时

OpenAI-compatible 和 Ollama 默认都使用流式响应，模型内容会逐块打印到终端，并在完成后合并为完整 JSON 进行校验。

- `--timeout`：单次网络操作超时秒数，默认 `900`；模型首个响应或后续数据块长期无响应时会触发。
- `--max-retries`：首次调用失败后的最大重试次数，默认 `2`。

流式生成已经输出部分内容后若发生超时，重试时终端可能再次显示完整的新响应；最终只保存通过 JSON Schema 校验的那一次结果。
