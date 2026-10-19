"""修改：文本状态。正式流程共用的三态契约，兼容事件字符串与列表。"""
WITH_EVENTS = 'AVAILABLE_WITH_EVENTS'
NO_EVENTS = 'AVAILABLE_NO_EVENTS'
UNAVAILABLE = 'UNAVAILABLE'


def resolve_text_status(events, text_status=None):
    has_events = bool(events)
    if isinstance(events, str):
        has_events = events.strip() not in ('', '【重大事件】')
    status = text_status or (WITH_EVENTS if has_events else UNAVAILABLE)
    if status not in (WITH_EVENTS, NO_EVENTS, UNAVAILABLE):
        raise ValueError(f'未知文本状态: {status}')
    if has_events != (status == WITH_EVENTS):
        raise ValueError('文本状态与事件内容不一致')
    return status


def text_instruction(status):
    if status == UNAVAILABLE:
        return ('文本通道状态：UNAVAILABLE。数据不可用，不代表没有事件。'
                '不得虚构新闻、政策或事件，不得将历史事件视为当前事实。'
                '反思不得生成 TEXTUAL/TEXT 证据，key_event_ids 必须为空；'
                '结论和失效条件不得补充事件原因。')
    if status == NO_EVENTS:
        return '文本通道状态：AVAILABLE_NO_EVENTS。数据覆盖正常，未发现符合筛选条件的重要事件，不代表没有风险。'
    return '文本通道状态：AVAILABLE_WITH_EVENTS。仅使用预测时点之前已获得的事件。'


def has_text_dependency(memory):
    return bool(memory.get('key_event_ids')) or any(
        ev.get('feature_layer') in ('TEXTUAL', 'TEXT')
        for ev in memory.get('evidence_units', []))


def validate_text_reflection(review, status):
    if status == UNAVAILABLE and has_text_dependency(review['experience_summary']):
        raise ValueError('文本不可用但反思包含文本证据，禁止入库')


def read_optional_text_data(path):
    import pandas as pd
    try:
        return pd.read_feather(path)
    except FileNotFoundError:
        # 修改：文本状态，缺失文本文件允许纯数值流程继续；其他读取错误照常抛出。
        return pd.DataFrame({'trade_date': pd.Series(dtype='datetime64[ns]')})
