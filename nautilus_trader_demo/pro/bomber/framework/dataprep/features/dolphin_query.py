"""只读适配的会话生命周期；不在导入或构造时加载SDK。"""
from threading import RLock

from ..sources.dolphindb import _session_factory
from .base import FeatureSourceError


class DolphinDbReadSession:
    def __init__(self, connection, *, session_factory=None):
        self.connection = connection
        self._factory = session_factory or _session_factory
        self._session = None
        self._lock = RLock()

    def open(self):
        with self._lock:
            if self._session is not None:
                return
            session = None
            try:
                c = self.connection
                session = self._factory(enableSSL=c.enable_ssl)
                if session.connect(c.host, c.port, c.username, c.password, reconnect=False,
                    readTimeout=c.read_timeout_seconds, writeTimeout=c.read_timeout_seconds) is not True:
                    raise FeatureSourceError("连接未确认成功")
                self._session = session
            except Exception:
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        pass
                raise FeatureSourceError("DolphinDB只读连接失败；检查网络/SDK/配置") from None

    def run(self, script):
        with self._lock:
            if self._session is None:
                raise FeatureSourceError("须先open DolphinDB只读来源")
            try:
                return self._session.run(script)
            except Exception:
                raise FeatureSourceError("DolphinDB只读查询失败；不解释为空数据") from None

    def close(self):
        with self._lock:
            session, self._session = self._session, None
            if session is not None:
                try:
                    session.close()
                except Exception:
                    raise FeatureSourceError("DolphinDB只读会话关闭失败") from None
