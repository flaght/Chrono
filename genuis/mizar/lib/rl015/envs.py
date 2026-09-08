import numpy as np
import pandas as pd
import gym, random, pdb
from gym import spaces
from typing import List, Dict, Any


class TradingEnv(gym.Env):
    """
    基于未来收益标签评分的连续动作强化学习环境 (rl012版本)。
    核心逻辑：
    1. SAC 每步输出三个 [-1, 1] 内的连续动作值，作为 Softmax 的输入。
    2. Softmax 得到 [中性, 向上, 向下] 连续权重，不是校准后的胜率。
    3. net_er_out = p_up - p_down，无观望门控、开仓门槛或换向惩罚。
    4. 奖励 = net_er_out * (future_ret_h / 品种尺度) * reward_scale。
    nxt1_ret[T] 已是 log(P[T+6]/P[T+1])，直接评分，不再累计或转换。
    holding_period=5 描述已有标签跨度，不是环境中的持仓状态。
    输入严格为1分钟数据；交易规则和实际仓位由外部策略负责。
    """

    def _build_train_windows_by_code(self) -> Dict[str, List[tuple]]:
        """在品种内按连续分钟段、连续有效标签段生成 [start, end) 窗口。"""
        if not self.asset_codes:
            raise ValueError("数据为空，无法创建品种 episode")
        if self.mode == "train" and self.train_scheme not in {"full", "half", "holding"}:
            raise ValueError("train_scheme 必须为 full、half 或 holding")
        if self.mode == "train" and self.asset_sampling != "balanced":
            raise ValueError("当前仅支持 asset_sampling='balanced'")
        windows_by_code = {}
        for code, positions in self._code_positions.items():
            valid = np.isfinite(self.future_ret_h[positions])
            if not valid.any():
                raise ValueError(f"{code} 没有有效 nxt1_ret 标签，无法创建 episode")
            windows = []
            for segment_left, segment_right in self._segments_by_code[code]:
                # NaN/Inf 标签也断开评分区间，不把无效标签补零后用于训练。
                mask = valid[segment_left:segment_right]
                edges = np.diff(np.r_[False, mask, False].astype(np.int8))
                for left, right in zip(np.flatnonzero(edges == 1),
                                       np.flatnonzero(edges == -1)):
                    left, right = int(left + segment_left), int(right + segment_left)
                    length = right - left
                    width = min(self.max_episode_steps, length) if self.max_episode_steps > 0 else length
                    if self.train_scheme == "half":
                        stride = max(1, width // 2)
                    elif self.train_scheme == "holding":
                        stride = max(1, min(self.holding_period, width))
                    else:
                        stride = width
                    if self.max_episode_steps == 0:
                        stride = width
                    # full 不重叠，保留不足一个完整窗口的尾段；其他方案允许重叠。
                    for start in range(left, right, stride):
                        end = min(start + width, right)
                        windows.append((start, end))
                        if end == right:
                            break
            windows_by_code[code] = windows
        return windows_by_code

    def _next_train_episode(self):
        # 【多资产修改 6：均衡抽样】均衡的是 episode 次数，不保证分钟条数
        # 或 Replay Buffer 中样本数量严格相等（窗口长度可能不同）。
        # 例如每轮 RB/HC 各一个窗口，并非先耗尽 RB 的全部历史再开始 HC。
        # balanced: 每轮每个品种恰好出现一次，轮内顺序随机。
        if self.train_asset_cursor >= len(self.train_asset_order):
            self._reshuffle_asset_order()
        code = self.train_asset_order[self.train_asset_cursor]
        self.train_asset_cursor += 1

        order = self.train_window_orders[code]
        cursor = self.train_window_cursors[code]
        if cursor >= len(order):
            order = list(range(len(self.train_windows_by_code[code])))
            if len(order) > 1:
                self.np_random.shuffle(order)
            self.train_window_orders[code] = order
            cursor = 0
        window = self.train_windows_by_code[code][order[cursor]]
        self.train_window_cursors[code] = cursor + 1
        return code, window[0], window[1]

    def _reset_train_sampling(self):
        # 各品种窗口独立打乱；某个品种的窗口用完后才重新洗牌。
        self.train_window_orders = {}
        self.train_window_cursors = {}
        for code, windows in self.train_windows_by_code.items():
            order = list(range(len(windows)))
            if len(order) > 1:
                self.np_random.shuffle(order)
            self.train_window_orders[code] = order
            self.train_window_cursors[code] = 0
        self._reshuffle_asset_order()

    def _reshuffle_asset_order(self):
        self.train_asset_order = list(self.asset_codes)
        if len(self.train_asset_order) > 1:
            self.np_random.shuffle(self.train_asset_order)
        self.train_asset_cursor = 0

    def _build_future_horizon_returns(self) -> np.ndarray:
        """
        直接读取预计算标签 nxt1_ret[T] = log(P[T+6] / P[T+1])。
        不求和、不平移、不做 expm1 转换，也不按 holding_period 裁掉尾部。
        上游负责收益端点、跨时段处理和跨数据集泄漏检查；无效标签应为NaN。
        环境只过滤非有限值。仅凭当前特征行无法验证标签使用的未来成交价格。
        """
        # 【多资产修改 8：标签归属】每行标签已属于该行code，无跨品种聚合。
        if "nxt1_ret" not in self.df.columns:
            raise ValueError("必须提供预计算的 nxt1_ret 五分钟log收益标签")
        ret = pd.to_numeric(self.df["nxt1_ret"],
                            errors="coerce").to_numpy(dtype=np.float64,
                                                      copy=True)
        ret[~np.isfinite(ret)] = np.nan
        return ret

    def _build_segment_start_offsets(self) -> np.ndarray:
        """记录每行所在连续分钟段在该品种内的起点，供  防跨时段取窗。"""
        segment_start = np.zeros(len(self.df), dtype=np.int64)
        self._segments_by_code: Dict[str, List[tuple]] = {
            code: [(0, len(positions))]
            for code, positions in self._code_positions.items()
        }
        if "trade_time" not in self.df.columns:
            raise ValueError("必须提供 trade_time 以验证1分钟频率及预测终点")
        raw_time = self.df["trade_time"]
        if pd.api.types.is_numeric_dtype(raw_time.dtype):
            raise ValueError("请先显式将数字 trade_time 转为 datetime，避免猜测时间单位")
        parsed_time = pd.to_datetime(raw_time, errors="raise")
        if parsed_time.isna().any():
            raise ValueError("trade_time 不允许缺失")
        for code, positions in self._code_positions.items():
            gaps = parsed_time.iloc[positions].diff().dt.total_seconds(
            ).to_numpy()
            if np.any(gaps[1:] <= 0):
                raise ValueError(f"{code} 存在重复或逆序时间；请先按品种时间排序并核查重复")
            starts = [0] + (np.flatnonzero(gaps[1:] != 60.0) + 1).tolist()
            boundaries = starts + [len(positions)]
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                segment_start[positions[left:right]] = left
            self._segments_by_code[code] = list(
                zip(boundaries[:-1], boundaries[1:]))
        return segment_start

    def _log_reset_window(self):
        if not self.debug_reset_log:
            return
        if self.reset_count % self.debug_reset_log_every != 0:
            return

        start_idx = int(self.active_positions[self.episode_start_offset])
        end_offset = self.episode_end_offset_exclusive
        window_len = int(max(0, end_offset - self.episode_start_offset))
        start_time = self.df.iloc[start_idx].get("trade_time", start_idx)
        end_label_idx = int(self.active_positions[max(0, end_offset - 1)])
        end_time = self.df.iloc[end_label_idx].get("trade_time", end_label_idx)

        if self.mode == "train":
            print(
                "[ENV_RESET][train] reset={0} code={1} offset=[{2},{3}) len={4} "
                "time=[{5} -> {6}]".format(self.reset_count, self.active_code,
                                           self.episode_start_offset,
                                           end_offset, window_len, start_time,
                                           end_time))
        else:
            print("[ENV_RESET][{0}] reset={1} code={2} full_eval=True "
                  "offset=[{3},{4}) len={5} time=[{6} -> {7}]".format(
                      self.mode, self.reset_count, self.active_code,
                      self.episode_start_offset, end_offset, window_len,
                      start_time, end_time))

    def _get_obs(self):
        if not self.use_tcn:
            return self._feature_values[self.current_step].astype(np.float32,
                                                                  copy=False)

        # 取当前品种截至 current_step 的最近 lookback 个分钟截面。
        # 不足部分用该品种最早可见的一行左填充，避免引入“全零=开盘”的伪信号。
        code = self._codes[self.current_step]
        code_positions = self._code_positions[code]
        code_offset = int(self._position_in_code[self.current_step])
        # 【多资产修改 9：时序模型 隔离】回看限制在同品种当前连续时段内。
        # 可以读取 episode 起点之前已发生的历史，但不会越过时段起点；
        # 不足 lookback 的部分复制最早可见行，不使用未来行补齐。
        segment_start = int(self._segment_start_in_code[self.current_step])
        start_offset = max(segment_start, code_offset - self.lookback + 1)
        positions = code_positions[start_offset:code_offset + 1]
        obs = self._feature_values[positions]
        if len(obs) < self.lookback:
            pad = np.repeat(obs[:1], self.lookback - len(obs), axis=0)
            obs = np.concatenate([pad, obs], axis=0)
        obs = obs.astype(np.float32, copy=False)
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def __init__(self, df: pd.DataFrame, features: List[str],
                 config: Dict[str, Any]):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = features
        self.config = config

        self.env_config = config.get("env_config", {})
        self.signal_config = config.get("signal_config", {})

        self.holding_period = int(self.env_config["holding_period"])
        self.use_tcn = bool(self.env_config.get("use_tcn", False))

        self.lookback = max(1, int(
            self.env_config["default_lookback"])) if self.use_tcn else 1
        self.reward_scale = float(self.env_config["reward_scale"])

        self.mode = str(self.env_config["mode"]).strip().lower()

        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(3, ),
                                       dtype=np.float32)

        observation_shape = ((self.lookback,
                              len(self.features)) if self.use_tcn else
                             (len(self.features), ))

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=observation_shape,
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(3, ),
                                       dtype=np.float32)

        self._feature_values = self.df[self.features].apply(
            pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self._feature_values = np.nan_to_num(self._feature_values,
                                             nan=0.0,
                                             posinf=0.0,
                                             neginf=0.0)

        # 【多资产修改 1：品种索引】原来直接沿整张 df 逐行训练，现在按 code
        # 找到各品种的行号。例如交错数据中 RB=[0,2,4]、HC=[1,3,5]。
        # positions 是全表行号，offset 是品种内部位置，两者不能混用。
        # 本段不排序；调用方必须保证每个 code 内部已按时间递增排列。

        self._codes = self.df["code"].astype(str).to_numpy()
        self._code_positions = {
            code: np.flatnonzero(self._codes == code)
            for code in pd.unique(self._codes)
        }

        self._position_in_code = np.empty(len(self.df), dtype=np.int64)
        for positions in self._code_positions.values():
            self._position_in_code[positions] = np.arange(len(positions),
                                                          dtype=np.int64)
        self.asset_codes = list(self._code_positions.keys())
        self.n_assets = len(self.asset_codes)
        self.sequence_max_gap_minutes = 1.0

        # 【多资产修改 2：连续时段】在每个品种内部识别时间断点，供训练
        # episode 和 TCN 历史窗口使用；缺分钟即断段。
        # 未来标签已由上游构造，不能根据输入窗口尾部缺行再次裁剪标签。
        self._segment_start_in_code = self._build_segment_start_offsets()

        self.current_step = 0
        self.future_ret_h = self._build_future_horizon_returns()

        self._valid_count_by_code = {}
        for code, positions in self._code_positions.items():
            valid_offsets = np.flatnonzero(
                np.isfinite(self.future_ret_h[positions]))
            # 这是最后一个有效预测标签之后的位置，不是 finite 数量；交易时段
            # 末尾可能存在分散的 NaN，不能用 count 截断整个品种序列。
            self._valid_count_by_code[code] = (int(valid_offsets[-1]) + 1 if
                                               len(valid_offsets) > 0 else 0)

        # 【多资产修改 3：收益尺度】RB、HC 分别除以各自固定的收益标准差。
        # train_model 会把训练尺度传给验证环境并保存给预测使用。
        # 验证/测试必须提供所有品种的训练尺度，不从未来样本估计。
        configured_scales = self.env_config.get("ret_scale_by_code")
        self.ret_scale_by_code: Dict[str, float] = {}
        for code, positions in self._code_positions.items():
            if isinstance(configured_scales,
                          dict) and code in configured_scales:
                scale = float(configured_scales[code])
            else:
                if self.mode != "train":
                    raise ValueError(
                        f"验证/测试缺少品种 {code} 的训练尺度 ret_scale_by_code")
                code_future = self.future_ret_h[positions]
                code_future = code_future[np.isfinite(code_future)]
                scale = float(
                    np.std(code_future)) if len(code_future) > 0 else 0.0
            if not np.isfinite(scale) or scale <= 1e-8:
                scale = 1e-8
            self.ret_scale_by_code[code] = scale

        self.history = []

        self.max_episode_steps = int(self.env_config["max_episode_steps"])
        if self.max_episode_steps < 0:
            self.max_episode_steps = 0

        self.softmax_temperature = float(
            self.env_config["softmax_temperature"])
        self.train_scheme = str(
            self.env_config["train_scheme"]).strip().lower()
        self.asset_sampling = str(
            self.env_config.get("asset_sampling", "balanced")).strip().lower()

        # 【多资产修改 4：两层调度】先选品种，再取该品种自己的训练窗口。
        # 品种顺序与窗口顺序分别维护，避免在全表切窗时混入另一品种。
        # 当前仅实现 balanced 调度，asset_sampling 没有其他分支。
        self.train_windows_by_code = self._build_train_windows_by_code()
        self.train_window_orders: Dict[str, List[int]] = {}
        self.train_window_cursors: Dict[str, int] = {}
        self.train_asset_order: List[str] = []
        self.train_asset_cursor = 0
        self.eval_asset_cursor = 0

        self.active_code = self.asset_codes[0]
        self.active_positions = self._code_positions[self.active_code]
        self.episode_start_offset = 0
        self.episode_end_offset_exclusive = 1
        self.current_code_offset = 0
        self.seed(seed=42)

        self.reset_count = 0
        self.debug_reset_log = True
        self.debug_reset_log_every = 5

        if self.mode == "train":
            self._reset_train_sampling()
            if self.debug_reset_log:
                print("[ENV_INIT][train] scheme={0} asset_sampling={1} "
                      "max_episode_steps={2} assets={3} windows={4}".format(
                          self.train_scheme, self.asset_sampling,
                          self.max_episode_steps, ",".join(self.asset_codes), {
                              k: len(v)
                              for k, v in self.train_windows_by_code.items()
                          }))

        elif self.debug_reset_log:
            print("[ENV_INIT][{0}] asset_episodes={1}".format(
                self.mode, ",".join(self.asset_codes)))

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        if self.mode == "train":
            code, start_offset, end_offset = self._next_train_episode()
        else:
            # 校验/测试每次 reset 顺序切换品种，每个 episode 完整覆盖一个品种。
            code = self.asset_codes[self.eval_asset_cursor % self.n_assets]
            self.eval_asset_cursor += 1
            start_offset = 0
            end_offset = int(
                np.isfinite(
                    self.future_ret_h[self._code_positions[code]]).sum())

        # 【多资产修改 7：绑定 episode】本次 reset 后 active_code 固定，
        # 所有 observation、reward 和 next_state 都沿 active_positions 取值。
        # 训练按连续区间切窗；验证/测试只评分有效标签，窗口仍按原始时间索引。
        self.active_code = code
        self.active_positions = self._code_positions[code]
        if self.mode != "train":
            self.active_positions = self.active_positions[np.isfinite(
                self.future_ret_h[self.active_positions])]
        self.episode_start_offset = int(start_offset)
        self.episode_end_offset_exclusive = int(end_offset)
        self.current_code_offset = self.episode_start_offset
        self.current_step = int(
            self.active_positions[self.current_code_offset])

        self.reset_count += 1
        self._log_reset_window()
        self.history = []
        return self._get_obs()

    def step(self, action: np.ndarray):
        raw_action = action.astype(float)
        if not np.isfinite(raw_action).all():
            raw_action = np.zeros(3, dtype=float)

        # 三个连续动作值经缩放后作为 Softmax 输入，输出仍为连续权重。
        # 此处 softmax_temperature 是乘数：正值越大，权重越集中；
        # 不同于常见的除以温度写法。权重集中不代表预测胜率更准确。
        temperature_scaled_action = raw_action * self.softmax_temperature
        exp_action = np.exp(temperature_scaled_action -
                            np.max(temperature_scaled_action))  # 减最大值防止溢出
        softmax_probs = exp_action / np.sum(exp_action)

        # 【纯预测】中性权重通过 Softmax 自然压低信号强度，不做开仓门控。
        # confidence 表示预测信号的绝对幅度，不是胜率。
        # er_value = float(softmax_probs[1] - softmax_probs[2])
        # confidence = abs(er_value)

        # 【预测门控】中性相对占优时输出0，不要求其中性权重超过50%。
        # 使用严格大于：与方向权重并列时仍计算连续多空差，不量化为+1/-1。
        # 这是预测输出定义，不表示实际开平仓。confidence统一为最终信号幅度。
        p_neutral, p_up, p_down = map(float, softmax_probs)
        is_neutral = p_neutral > p_up and p_neutral > p_down
        er_value = 0.0 if is_neutral else p_up - p_down
        confidence = abs(er_value)

        raw_action_str = f"[{raw_action[0]:.4f},{raw_action[1]:.4f},{raw_action[2]:.4f}]"
        soft_action_str = f"[{softmax_probs[0]:.4f},{softmax_probs[1]:.4f},{softmax_probs[2]:.4f}]"

        future_ret_h = float(self.future_ret_h[self.current_step])
        if not np.isfinite(future_ret_h):
            raise ValueError("当前样本无有效未来标签，不应进入评分 episode")
        net_er_out = er_value  # 预测信号，不是实际仓位
        active_count = int(er_value != 0)

        # current_ret 日志字段也记录当前行的5分钟log标签，不是1分钟收益。
        nxt1_ret = future_ret_h

        target_ret_raw = future_ret_h if np.isfinite(future_ret_h) else 0.0
        # target_ret = target_ret_raw
        # 依据当前 episode 品种选择尺度；这是按品种波动调整后的收益目标。
        ret_scale = self.ret_scale_by_code[self.active_code]
        target_ret = target_ret_raw / ret_scale

        step_reward = net_er_out * target_ret

        if not np.isfinite(step_reward):
            step_reward = 0.0

        scaled_reward = step_reward * self.reward_scale

        trade_time = self.df.iloc[self.current_step].get(
            'trade_time', self.current_step)

        # direction 仅记录预测信号的符号，不表示已执行交易。
        direction = 1 if er_value > 0 else (-1 if er_value < 0 else 0)
        signal = er_value

        self.history.append({
            'trade_time': trade_time,
            'code': self.active_code,
            'raw_action': raw_action_str,
            'soft_action': soft_action_str,
            'signal': signal,
            'label_valid': True,
            'direction': direction,
            'confidence': confidence,
            'neutral_weight': p_neutral,
            'is_neutral': is_neutral,
            'net_er_out': net_er_out,
            'er_value': er_value,
            'active_signals': active_count,
            'current_ret': nxt1_ret,
            'target_ret_raw': target_ret_raw,
            'target_ret': target_ret,
            'ret_scale': ret_scale,
            'reward': step_reward,
            'reward_scaled': scaled_reward,
            'future_ret_h': future_ret_h,
            # 将 trade_cost 设为 0（因做纯因子预测时不在此纳入交易摩擦）
            'trade_cost': 0.0
        })

        # 【多资产修改 10：状态转移】原来是全表 current_step += 1；现在只
        # 增加品种内 offset，再映射回全表。RB 当前行的 next_state 仍为 RB。
        # 到窗口末尾返回 done=True，保留当前观测作为终止观测；下个品种由
        # reset 选择，不把品种切换写成一条未终止的 transition。
        next_code_offset = self.current_code_offset + 1
        done = next_code_offset >= self.episode_end_offset_exclusive
        if not done:
            self.current_code_offset = next_code_offset
            self.current_step = int(
                self.active_positions[self.current_code_offset])

        return self._get_obs(), scaled_reward, done, {}
