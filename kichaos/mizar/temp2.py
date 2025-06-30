import pandas as pd
import numpy as np
from collections import deque

# 假设 process_features_for_window 和其他辅助函数已定义
# ... (您的 process_features_for_window 代码) ...


class LiveFeatureProcessor:
    def __init__(self, factor_cols):
        self.factor_cols = factor_cols
        
        # 1. 定义窗口大小 (关键参数)
        self.correction_window_size = 180 * 360  # 64,800
        self.normalization_window_size = 60

        # 2. 初始化数据缓冲区
        # 修正缓冲区：存储原始因子数据，用于生成修正规则
        self.raw_feature_buffer = deque(maxlen=self.correction_window_size)
        # 标准化缓冲区：存储修正后的因子数据，用于滚动标准化
        self.corrected_feature_buffer = deque(maxlen=self.normalization_window_size)

        # 3. 缓存修正规则 (性能优化的核心)
        # self.correction_rules 是一个字典，key是因子名，value是该因子的处理规则
        # e.g., {'factor_A': {'action': 'log1p', 'fillna': 0.5}, ...}
        self.correction_rules = None
        self.rules_update_counter = 0
        # 每隔多少个bar更新一次规则，例如每隔一天 (360个bar)
        self.UPDATE_RULES_FREQUENCY = 360 

    def is_ready(self):
        """检查处理器是否已预热完毕，可以产出有效特征"""
        return len(self.raw_feature_buffer) == self.correction_window_size

    def _update_correction_rules(self):
        """
        [计算密集型操作]
        使用整个修正缓冲区的数据运行 `process_features_for_window` 来更新规则。
        """
        print(f"[{pd.Timestamp.now()}] Updating feature correction rules...")
        # 1. 将deque转为DataFrame
        df_window = pd.DataFrame(list(self.raw_feature_buffer), columns=self.factor_cols)
        
        # 2. 运行您的研究函数，但我们只关心生成的'report'
        _, report = process_features_for_window(df=df_window, factor_cols=self.factor_cols)
        
        # 3. 提取并缓存核心规则，以便快速应用
        # 这里需要您根据 report 结构，设计一个更简洁的 rules 字典
        # 这是一个示例，您需要根据 report 的具体内容来编写
        rules = {}
        for factor_name, factor_report in report.items():
            # 简化版规则提取，您需要根据自己的需求扩展
            rules[factor_name] = {
                'fillna_value': factor_report['initial_stats_summary']['median'], # 假设用中位数填充
                'transformation': factor_report['transformation_applied'],
                # 如果是 boxcox/yeojohnson，还需要保存lambda值，这需要修改 process_features_for_window 来返回它
                # 'lambda': factor_report.get('lambda', None) 
            }
        self.correction_rules = rules
        print("Rules updated successfully.")

    def _apply_rules_to_bar(self, raw_bar_series):
        """
        [轻量级操作]
        将缓存的规则应用到单行新数据上。
        """
        corrected_bar = raw_bar_series.copy()
        for factor_name in self.factor_cols:
            rules = self.correction_rules[factor_name]
            
            # 步骤A: 填充缺失值
            if pd.isna(corrected_bar[factor_name]):
                corrected_bar[factor_name] = rules['fillna_value']

            # 步骤B: 应用转换
            # 这是一个简化的示例，您需要根据您的`apply_transformation`函数来完整实现
            transform_str = rules['transformation']
            if 'log1p' in transform_str:
                # 注意处理反射（reflection）的情况
                if '(R)' in transform_str:
                    # 假设您的反射是围绕某个中心值，或者简单取负。这里以取负为例
                    # 在实际应用中，您需要知道反射的具体操作才能正确反转
                    # 更好的方法是在`_update_correction_rules`时保存反射中心点
                    value = -corrected_bar[factor_name]
                    transformed_value = np.log1p(value)
                else:
                    transformed_value = np.log1p(corrected_bar[factor_name])
                corrected_bar[factor_name] = transformed_value
            elif 'sqrt' in transform_str:
                # ... 实现 sqrt 逻辑 ...
                pass
            elif 'boxcox' in transform_str:
                # lmbda = rules['lambda']
                # ... 实现 boxcox 逻辑 ...
                pass
            # ... 其他转换逻辑 ...

        return corrected_bar

    def process_new_bar(self, raw_bar_series):
        """
        处理单个新bar的主函数。
        raw_bar_series: 一个Pandas Series，包含最新的原始因子值。
        """
        # 步骤 1: 更新原始数据缓冲区
        self.raw_feature_buffer.append(raw_bar_series[self.factor_cols])
        self.rules_update_counter += 1

        # 预热阶段：如果缓冲区未满，则不进行任何处理
        if not self.is_ready():
            print(f"Warming up... Correction buffer: {len(self.raw_feature_buffer)}/{self.correction_window_size}")
            return None

        # 步骤 2: 周期性地更新修正规则
        if self.rules_update_counter >= self.UPDATE_RULES_FREQUENCY or self.correction_rules is None:
            self._update_correction_rules()
            self.rules_update_counter = 0 # 重置计数器

        # 步骤 3: 对新来的bar应用已缓存的修正规则
        latest_corrected_bar = self._apply_rules_to_bar(raw_bar_series)
        
        # 步骤 4: 更新标准化缓冲区
        self.corrected_feature_buffer.append(latest_corrected_bar)

        # 预热阶段：如果标准化缓冲区未满，依然不能产出最终结果
        if len(self.corrected_feature_buffer) < self.normalization_window_size:
            print(f"Warming up... Normalization buffer: {len(self.corrected_feature_buffer)}/{self.normalization_window_size}")
            return None

        # 步骤 5: 执行滚动标准化
        # 将deque转为DataFrame以方便计算
        df_norm_window = pd.DataFrame(list(self.corrected_feature_buffer), columns=self.factor_cols)
        
        rolling_mean = df_norm_window.mean()
        rolling_std = df_norm_window.std().replace(0, 1e-6) # 与研究代码一致

        # 对【当前修正后】的bar进行标准化
        final_bar = (latest_corrected_bar - rolling_mean) / rolling_std

        return final_bar


historical_data_stream = None
all_factor_names = None
live_data_stream = None
# --- 初始化 ---
# 假设 all_factor_names 是你100个特征的列名列表
processor = LiveFeatureProcessor(factor_cols=all_factor_names)

# --- 模拟实盘数据流 ---
# 你需要先用至少 64,800 条历史数据来“填满”缓冲区
for historical_bar in historical_data_stream:
    processor.process_new_bar(historical_bar) # 在预热阶段，这将返回 None

print("Processor is now fully warmed up and ready for live trading.")

# --- 进入真正的实时处理循环 ---
for new_live_bar in live_data_stream:
    final_features = processor.process_new_bar(new_live_bar)
    
    if final_features is not None:
        # `final_features` 就是可以输入到你模型的最终特征向量
        # model.predict(final_features)
        print("Generated final features for model input:")
        print(final_features)