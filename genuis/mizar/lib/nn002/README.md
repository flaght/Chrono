# Transformer-MSE 多资产基线

这是独立于 `rl015`、`rl016` 和 `tcn_mse` 的监督学习版本，用来与
TCN-MSE 做严格的同条件对照。数据、标签、收益标准化、抽样与 IC 选模均
保持一致，核心差别只有时序网络改为因果 Transformer Encoder。

## 核心定义

```text
target_z       = nxt1_ret_5h / train_std(code)
predicted_z    = 3.0 * tanh(model_output)
train_loss     = mean((predicted_z - target_z)^2)
selection      = min(RB rolling_IC_mean, HC rolling_IC_mean)
rolling_IC     = minute % 5 == 0 后 rolling(15, min_periods=5).corr
```

- RB、HC 分别使用自身训练集收益标准差；不减均值，不使用校验或测试数据
  拟合尺度。
- 每个训练 batch 尽可能各含一半 RB 与 HC，避免样本更多的品种支配模型。
- 输入是 `(batch, 20, 24)`；窗口不会跨品种或不连续分钟断点。
- Encoder 使用严格的因果注意力遮罩，末端 token 只能看到 T 及之前数据。
- `predicted_z` 是标准化收益预测，`predicted_ret_h` 是还原后的收益率预测。
- 最佳模型只按完整校验集最弱品种 IC 选择，测试集不参与选模。
- 默认在第 5、10、15、20 个 epoch 验证，共4个候选版本，与 TCN-MSE
  的默认候选数量一致。

## 为什么先使用轻量结构

当前只有24个特征、20分钟窗口，默认 `d_model=64`、2层、4头已经足够做
有效对照。Transformer 参数过大更容易在 RB/HC 两个品种上过拟合，也会
显著降低训练和全校验集推理速度。只有当它稳定优于 TCN-MSE 后，再尝试
增加 `d_model`、层数或 lookback。

## 文件与输出

- `model.py`：带可学习位置编码和因果遮罩的 Transformer 回归器。
- `data.py`：按需构造时序窗口与 RB/HC 均衡批次。
- `metrics.py`：与当前因子评估一致的5分钟抽样滚动 IC。
- `train.py`：MSE 训练、完整校验 IC 选模与早停。
- `predict.py`：最佳模型的测试集批量推理。
- `rbb.yaml`：推荐起始参数。
- 调用端：`6.1.7.build_transformer_mse_strategy.py`。

```text
rl/transformer_mse/result/<参数标签>/
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

部署到服务器时，将 `transformer_mse` 放到项目 `lib/` 下，将调用端放到
`z06_models/` 下；训练和预测继续使用项目原来的 `Tactix` 参数进入方式。
