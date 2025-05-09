import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch, pdb, os
from ultron.optimize.wisem import *


class Base(Dataset):

    @classmethod
    def generate(cls,
                 data,
                 features,
                 window=1,
                 target=None,
                 start_date=None,
                 seq_cycle=0,
                 time_name='trade_time',
                 time_format='%Y-%m-%d %H:%M:%S'):
        names = []
        res = []
        start_time = data[time_name].min().strftime(
            time_format) if start_date is None else start_date
        data_raw = data.set_index([time_name, 'code']).sort_index()
        raw = data_raw[features]
        uraw = raw.unstack()
        for i in range(0, window):
            names += ["{0}_{1}d".format(c, i) for c in features]
            res.append(uraw.shift(i).loc[start_time:].stack())
        dt = pd.concat(res, axis=1)
        dt.columns = names
        dt = dt.reindex(data_raw.index)
        dt = dt.loc[start_time:].dropna()
        if target is not None:
            dt = dt.sort_index()
            dt1 = data.set_index([time_name, 'code'])[target].sort_index()
            dt = pd.concat([dt, dt1], axis=1)
            dt = dt.dropna().sort_index()

        return cls(dt,
                   features,
                   window=window,
                   seq_cycle=seq_cycle,
                   target=target,
                   time_format=time_format)

    def __init__(self,
                 data=None,
                 features=None,
                 window=None,
                 target=None,
                 seq_cycle=0,
                 time_name='trade_time',
                 time_format='%Y-%m-%d %H:%M:%S'):
        self.samples = []
        self.seq_cycle = seq_cycle
        self.start_pos = self.seq_cycle
        self.time_name = time_name
        if data is None:
            return
        self.data = data
        self.features = features
        wfeatures = [f"{f}_{i}d" for f in features for i in range(window)]
        self.window = window

        self.wfeatures = wfeatures  # 时间滚动特征
        self.sfeatures = self.features  # 原始状态特征
        self.code = self.data.index.get_level_values(1)
        self.trade_time = self.data.index.get_level_values(0).strftime(
            time_format)
        grouped = data.groupby('code')
        for code, group in grouped:
            # 按交易日期排序
            print(f"Processing code: {code}")
            sorted_group = group.sort_index(level=time_name)
            # 提取特征和标签
            #pdb.set_trace()
            time_index1 = sorted_group.index.get_level_values(0).strftime(
                time_format)
            features = sorted_group[self.wfeatures].values
            targets = sorted_group[target].values
            n_samples = len(sorted_group) - self.seq_cycle + 1
            #for i in range(n_samples):
            for index, i in enumerate(range(n_samples)):
                #x = features[i:i + self.seq_cycle]  # (5, 3)
                #y = targets[i + self.seq_cycle - 1]  # 取窗口最后一天对应的y值
                x = features[i - self.seq_cycle:i]
                y = targets[i - 1]
                if x.shape[0] != self.seq_cycle:
                    continue
                self.samples.append(
                    (time_index1[index - 1], torch.from_numpy(x),
                     torch.from_numpy(y)))


class Dataset10(Base):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 按trade_time分组存储样本
        self.time_groups = {}
        time_index = 0
        for t, x, y in self.samples:
            time_key = t
            if time_key not in self.time_groups:
                self.time_groups[time_key] = []
            self.time_groups[time_key].append((x, y))
            time_index += 1

        # 保存所有时间点键
        self.time_keys = list(self.time_groups.keys())

    def __len__(self):
        """返回总样本数(非时间点数)"""
        return sum(len(v) for v in self.time_groups.values())

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        Args:
            idx: 全局样本索引
        Returns:
            单个样本(values和target)
        """
        # 计算对应的时间点和局部索引
        count = 0
        for time_key in self.time_keys:
            group = self.time_groups[time_key]
            if idx < count + len(group):
                return {
                    'time': time_key,
                    'values':
                    group[idx - count][0],  # [seq_cycle, num_features]
                    'target': group[idx - count][1]  # 标量
                }
            count += len(group)
        raise IndexError("Index out of range")

    @staticmethod
    def collate_fn(batch):
        """
        自定义collate函数
        Args:
            batch: DataLoader获取的一批样本
        Returns:
            合并后的批次数据
        """
        # 自动按时间分组
        time_groups = {}
        for item in batch:
            time_key = item['time']
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(item)

        # 对每个时间点的数据单独处理
        results = []
        for time_key, items in time_groups.items():
            values = torch.stack([item['values'] for item in items
                                  ])  # [N, seq_cycle, num_features]
            targets = torch.stack([item['target'] for item in items])  # [N]
            results.append({
                'time': time_key,
                'values': values,
                'target': targets
            })

        return results