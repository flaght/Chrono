# -*- coding: utf-8 -*-
import torch


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    TRAIN_STEPS = 200
    MAX_FORMULA_LEN = 6

    # 时间序列算子默认周期列表
    DEFAULT_PERIODS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

