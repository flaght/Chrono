"""协作进程的物理账户单写保护；首期只支持同机 Linux/macOS flock。"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class AccountWriterIdentity:
    venue: str
    environment: str
    account: str

    def __post_init__(self):
        for name in ("venue", "environment", "account"):
            value = getattr(self, name).strip()
            if not value:
                raise ValueError("物理账户身份不能为空")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "venue", self.venue.upper())
        object.__setattr__(self, "environment", self.environment.lower())

    @property
    def key(self):
        # 不使用client_id、API key或前置地址：同账户不同通道必须争同一锁。
        value = json.dumps((self.venue, self.environment, self.account))
        return hashlib.sha256(value.encode()).hexdigest()


@contextmanager
def account_writer_lease(identities, *, lock_directory=Path("/tmp")):
    import fcntl
    identities = tuple(identities)
    keys = [item.key for item in identities]
    if len(set(keys)) != len(keys):
        raise ValueError("同一物理账户不能绑定多个活动写入客户端")
    with ExitStack() as resources:
        for key in sorted(keys):
            path = Path(lock_directory) / f"bomber-writer-{key}.lock"
            # 不删除锁文件；删除后会产生第二个inode并破坏互斥。
            handle = resources.enter_context(path.open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("物理账户已有活动写入所有者") from error
        yield
