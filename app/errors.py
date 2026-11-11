"""Typed errors carrying a machine-readable reason code.

Reason codes are split into two groups because the recovery controller treats
them differently (spec: missing weights, unavailable GPU access and unsupported
hardware need configuration correction, not repeated restarts).
"""
from __future__ import annotations

# Not recoverable by restarting the process: an operator must change config,
# restore device access, or investigate.
CONFIG_REASON_CODES = frozenset(
    {
        "config_invalid",
        "weights_missing",
        "weights_checksum_mismatch",
        "weights_corrupt",
        "model_manifest_missing",
        "inputs_missing",
        "inputs_checksum_mismatch",
        "cuda_unavailable",
        "cuda_check_failed",
        "unsupported_hardware",
    }
)

# Transient or runtime states.
TRANSIENT_REASON_CODES = frozenset(
    {"loading", "smoke_inference_failed", "worker_stuck", "worker_dead", "unknown"}
)


def is_config_reason(code: str | None) -> bool:
    return code in CONFIG_REASON_CODES


class LabError(Exception):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message
