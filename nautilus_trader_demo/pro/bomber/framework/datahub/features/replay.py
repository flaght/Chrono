"""用模拟时钟合并特征发布、行情与等待期限；不使用sleep或真实墙钟。"""
from dataclasses import dataclass
import heapq

from .contracts import FeatureError
from .frames import FeatureFrameAssembler


@dataclass(frozen=True)
class MarketFrameInput:
    frame_id: str
    event_ns: int
    available_ns: int
    market: object


def replay_feature_frames(hub, requirements, market_frames, records, *, wait_ns=2_000_000_000):
    assembler = FeatureFrameAssembler(hub, requirements, wait_ns=wait_ns)
    queue, sequence = [], 0
    def push(timestamp, kind, item):
        nonlocal sequence
        heapq.heappush(queue, (timestamp, kind, sequence, item))
        sequence += 1
    for record in records:
        push(record.visible_ns, 0, record)
    for frame in market_frames:
        if frame.available_ns < frame.event_ns:
            raise FeatureError("完整市场帧不能提前可见")
        push(frame.available_ns, 1, frame)
    outcomes = []
    while queue:
        now = queue[0][0]
        # 相同时点特征先发布，市场帧后注册，最后推进决策/超时。
        while queue and queue[0][0] == now:
            _, kind, _, item = heapq.heappop(queue)
            if kind == 0:
                hub.features.store.publish((item,))
            elif kind == 1:
                assembler.offer(item.frame_id, item.event_ns, item.available_ns, item.market)
                push(now + wait_ns, 2, None)
        outcomes.extend(assembler.poll(now))
    return tuple(outcomes)
