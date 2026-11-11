# RB-002 Model weights / inputs missing or corrupt

**Alerts:** `ServiceUnavailable`, `OperatorActionRequired` · **Reason codes:** `weights_missing`, `weights_checksum_mismatch`,
`weights_corrupt`, `model_manifest_missing`, `inputs_missing`, `inputs_checksum_mismatch`, `config_invalid` · **Drill:** F03

The service stays up but not ready and rejects `/infer` with `503`. The controller reports `operator_required` and **does not restart** it.

1. `curl -s 127.0.0.1:8000/readyz` → `reason_code` and `reason` (path and the first differing hash).
2. Check the mount: `ls -l manifests/weights/` and `sha256sum manifests/weights/*.pth` vs `manifests/model.json`.
3. Missing: `make weights` (explicit download into the pinned image). Mismatch/corrupt: delete the file, `make weights`; if the hash still differs, treat the source as untrusted and investigate before re-pinning `manifests/model.json`.
4. Inputs: `python -m preflight.container_checks` shows which input file fails; restore from version control, never regenerate silently (hashes would change).
5. Controlled redeploy: `make down-workload && make up-gpu`, then `make validate`.
6. Confirm `OperatorActionRequired` resolved and `/readyz` 200.
