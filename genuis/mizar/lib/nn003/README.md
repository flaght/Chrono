# HybridTransformer-MSE 多资产基线

该目录是全新、独立的监督学习实现。它使用现有依赖：

```python
from kichaos.nn.HybridTransformer import HybridTransformer
```

没有修改 `HybridTransformer` 源码，也没有导入 `rl015`、`rl016`、
`tcn_mse` 或 `transformer_mse` 的代码。

## 为什么使用 HybridTransformer 基类

`SequentialHybridTransformer` 在原实现内部固定使用 `sigmoid`，输出只能为
正数，不适合预测有正有负的未来收益率。因此，本版本使用导出的
`HybridTransformer` 基类作为 Encoder-Decoder 骨干，取最后一个分钟位置的
输出，再使用：

```text
predicted_z = prediction_bound_std * tanh(backbone_last_output)
```

默认输出范围为 `[-3, 3]`。

Encoder 显式传入三角因果遮罩，Decoder 自注意力使用原组件自带的因果
遮罩。输入窗口本身也只包含 T 及之前的数据，不能接触标签覆盖的 T+1～T+6。

## 与 TCN-MSE、Transformer-MSE 的可比条件

```text
target_z       = nxt1_ret_5h / train_std(code)
train_loss     = mean((predicted_z - target_z)^2)
selection      = min(RB rolling_IC_mean, HC rolling_IC_mean)
rolling_IC     = minute % 5 == 0 后 rolling(15, min_periods=5).corr
```

- RB、HC 各用自身训练集收益标准差，不减均值。
- 每个训练批次尽可能各包含一半 RB 和 HC。
- 20分钟窗口不跨品种或不连续分钟断点。
- 最佳模型只使用完整校验集选出，测试集不参与选模。
- 默认第5、10、15、20个epoch验证，保证候选模型数一致。

## 服务器放置

- 将整个 `hybrid_transformer_mse/` 放到服务器项目的 `lib/` 下。
- 将 `6.1.8.build_hybrid_transformer_mse_strategy.py` 放到 `z06_models/`。
- 参数使用本目录的 `rbb.yaml`。

输出目录：

```text
rl/hybrid_transformer_mse/result/<参数标签>/
  config.json
  model.log
  models/best_model.pt
  models/final_model.pt
  logs/validation_history.jsonl
  logs/final_full_ic_validation.json
  logs/final_rolling_ic.csv
  metrics/test_results.csv
  metrics/test_results_metrics.json
  metrics/test_results_rolling_ic.csv
```

HybridTransformer 同时运行 Encoder 和 Decoder，比简化版 Transformer 占用
更多显存。默认训练 `batch_size=1024`、推理 `4096`；确认 A100 显存余量后
可逐步提高到训练2048、推理8192。
