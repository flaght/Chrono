---
name: factor-mining-orchestrator
description: 编排因子想法研究、因子代码开发、受控运行验证及因子评估，维护可审计、可恢复的流水线状态。用于运行或继续完整因子挖掘流程；不复制子 Skill 的领域逻辑，不自动发布因子。
---

# 因子挖掘总控

把 `factor-idea-researcher`、`factor-development` 和 `factor-evaluation` 串成文件化流程。总控只负责路由、状态、产物校验和恢复，不替代子 Skill 的契约。

总控本身是确定性状态机，不使用提示词或大模型；只有被路由到 `python` 的研究、开发子阶段才调用各自环境配置的模型。状态拓扑和恢复规则统一定义在 [references/orchestration-contract.md](references/orchestration-contract.md)。

## 入口选择

每个需要大模型的阶段都必须显式记录 `execution: codex|python`：

- `codex`：当前 Codex 按对应子 Skill 直接工作；不得读取 `.env` 或调用环境变量中的外部模型。完成后把产物路径写入总控请求，再继续总控流程。
- `python`：总控运行子 Skill 的 Python 脚本；外部模型配置由对应阶段的 `env_file`、进程环境和脚本参数决定。
- `factor-evaluation` 是确定性脚本，不调用大模型。

用户未指定时，Codex 交互任务默认使用 `codex`；只有用户明确要求脚本、外部模型或自动流水线时才使用 `python`。

## 执行流程

1. 完整阅读 [references/orchestration-contract.md](references/orchestration-contract.md)。
2. 首次部署时编辑一次 [config/pipeline.defaults.json](config/pipeline.defaults.json)，配置测试数据、行情、收益、评估器路径和常用参数。
   若配置了 `research.feature_dictionary` 或 `development.feature_dictionary`，加载指定文件；否则研究阶段使用自身内置 Orion 字段字典，并把实际使用的字典产物自动传给开发阶段。
3. 日常 Python 全流程只需用 `--idea "..."` 提交短想法，或用 `--input idea.md|idea.txt` 提交较长文本；无需为每个想法手写请求 JSON。需要为本次运行指定最终代码因子名时增加 `--factor-name tf001`。
4. 需要覆盖单次参数时才使用 `--request`；请求内容覆盖默认配置。
5. Codex 模式下，分别按子 Skill 生成研究或开发产物，然后在请求中传入 `specification` 或 `validated_factor_artifact`；不要让总控脚本假装调用 Codex。
6. Python 模式只要默认配置提供了 `development.test_data`，候选代码生成后就自动运行 `finalize_validation.py`；验证成功立即进入绩效评估，不等待人工审批。
7. `development.approval` 仅用于可选审计，不是继续流程的条件。
8. 只有 `validated-factor-artifact.json` 的状态和哈希通过检查后，才进入评估；不自动调用发布流程。

## 状态规则

状态只允许向前推进：`initialized → researching → developing → awaiting_runtime_validation → validating → evaluating → completed`。Codex 门使用 `awaiting_codex_research` 或 `awaiting_codex_development`。失败记录为 `failed`，保留已经生成的阶段产物；修改请求后可在同一运行目录恢复。总控不得无限重试，模型阶段的重试次数由子 Skill 的有限 `max_retries` 控制。

返回时报告当前状态、下一步人工动作、各阶段入口和 `pipeline-manifest.json` 的绝对路径。
