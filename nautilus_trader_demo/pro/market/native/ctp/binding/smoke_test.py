"""No-network smoke test for the standalone bomber_ctp_md extension."""

from __future__ import annotations

import platform
import struct
import sys
import tempfile
from pathlib import Path

import bomber_ctp_md


def main() -> None:
    print("python:", sys.version.split()[0])
    print("platform:", platform.platform())
    print("machine:", platform.machine())
    print("pointer_bits:", struct.calcsize("P") * 8)
    print("module:", bomber_ctp_md.__file__)

    api = bomber_ctp_md.MdApi()
    print("ctp_api_version:", api.getApiVersion())

    with tempfile.TemporaryDirectory(prefix="bomber-ctp-md-") as directory:
        flow_path = f"{Path(directory).resolve()}/"
        api.createFtdcMdApi(flow_path, False)
        print("create_release: OK")
        api.release()

    # Guard behavior should raise Python exceptions rather than segfaulting.
    try:
        api.getTradingDay()
    except RuntimeError as exc:
        print("uninitialized_guard: OK", exc)
    else:
        raise AssertionError("getTradingDay should fail after release")

    print("native smoke test: OK")


if __name__ == "__main__":
    main()
