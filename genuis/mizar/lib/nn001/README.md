# TCN-MSE 多资产基线

该目录是一套不依赖 `rl016` 的监督学习基线，用来回答：在完全相同的
RB/HC 数据、TCN 结构、收益标准化和 IC 评估口径下，直接 MSE 回归是否
优于 SAC。

## 核心定义

```text
target_z       = nxt1_ret_5h / train_std(code)
predicted_z    = 3.0 * tanh(model_output)
train_loss     = mean((predicted_z - target_z)^2)
selection      = min(RB rolling_IC_mean, HC rolling_IC_mean)
rolling_IC     = 5分钟抽样后 rolling(15, min_periods=5).corr
```

- RB、HC 分别使用各自训练集收益标准差，校验集和测试集只复用训练尺度。
- 每个训练 batch 尽量各含一半 RB 和 HC，避免 RB 样本量较大而支配模型。
- TCN 窗口不会跨品种或不连续分钟断点。
- `predicted_z` 是标准化收益预测；`predicted_ret_h` 才是还原后的收益率。
- 最佳模型只由完整校验集最弱品种滚动 IC 选择，测试集不参与选模。
- 默认在第 5、10、15、20 个 epoch 做四次完整验证，与当前 SAC 的四次
  完整验证候选数量一致，减少因为挑选次数不同带来的比较偏差。

验证结果另外给出 `zero_prediction_mse` 和 `mse_skill_vs_zero`。后者大于
0 表示模型 MSE 优于永远预测零，当前 SAC 的对照值则为负数。
- 默认20个epoch、每5个epoch验证一次，共4个候选检查点，与当前SAC的
  4次完整验证数量保持一致。

## 文件

- `model.py`：因果 TCN 和直接回归头。
- `data.py`：低内存时序窗口、RB/HC 均衡批次。
- `metrics.py`：与现有因子评估一致的滚动 IC。
- `train.py`：MSE 训练、完整验证、IC 选模和早停。
- `predict.py`：最佳模型测试集批量推理。
- `rbb.yaml`：推荐起始参数。
- 调用端：`6.1.6.build_tcn_mse_strategy.py`。

## 输出

结果写入原任务目录下：

```text
rl/tcn_mse/result/<参数标签>/
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

训练和测试仍通过项目原有的 `Tactix` 参数进入方式运行，只需把服务器上的
调用脚本改为 `6.1.6.build_tcn_mse_strategy.py`，并使用本目录 YAML 中的参数。
