"""CTP配置与分钟Feed工厂；构造不连接，具体实现由调用方注入。"""

import os


def required(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"缺少环境变量: {name}")
    return value


def build_simnow_transport(transport_factory, *, client_id, flow_path, timeout_seconds):
    if required("CTP_BROKER_ID") != "9999":
        raise ValueError("本入口仅允许SimNow BrokerID=9999")
    investor = required("CTP_ACCOUNT_ID")
    return transport_factory(client_id=client_id, account_id=investor,
        front=required("CTP_TD_ADDRESS"), broker_id="9999", investor_id=investor,
        password=required("CTP_PASSWORD"), app_id=os.getenv("CTP_APP_ID", ""),
        auth_code=os.getenv("CTP_AUTH_CODE", ""), flow_path=os.getenv("CTP_TD_FLOW_PATH", flow_path),
        production_mode=os.getenv("CTP_PRODUCTION_MODE", "true").lower() in {"1", "true", "yes", "on"},
        timeout_seconds=timeout_seconds)


def build_md_config(config_factory, lifecycle, *, flow_path):
    return config_factory(front=lifecycle.md_front, broker_id="9999",
        user_id=lifecycle.transport.investor_id, password=lifecycle.transport.password,
        flow_path=os.getenv("CTP_MD_FLOW_PATH", flow_path),
        production_mode=lifecycle.transport.production_mode)


def build_minute_feed(source_id, upstream, policy, *, after_tick=None):
    from bomber.framework.market.stream import ReceiveTimeTradeTickBarFeed, TradeTickBarFeed
    factory = ReceiveTimeTradeTickBarFeed if policy.bar_time_basis == "receive" else TradeTickBarFeed
    return factory(source_id, upstream, after_tick=after_tick)
