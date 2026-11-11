"""Start / stop the separate `inference-fault` case used by F03 and F05.

Delegates to the active runtime: a Compose service in Docker mode, a supervised process in native
(Sol) mode. In native mode "no GPU device mapping" (F05) is expressed by hiding all devices from the
process (CUDA_VISIBLE_DEVICES=""), the closest process-level equivalent.
"""
from __future__ import annotations

from faults import runtime
from faults.runtime import GPU_OVERRIDE  # noqa: F401


def fault_case_up(backend: str, model_path: str, with_gpu: bool = False):
    return runtime.get().fault_case_up(backend, model_path, with_gpu)


def fault_case_down():
    return runtime.get().fault_case_down()
