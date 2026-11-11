# GPU Workload Bring-up, Validation and Observability Lab - entry points.
# Documented workflow: make preflight, up-gpu, validate, benchmark, fault-test, report, down.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PY        ?= .venv/bin/python
# PROFILE: gpu | cpu (workload under test for validate / fault-test / soak)
PROFILE   ?= gpu
IMAGE     ?= gpu-lab/app:local
export LAB_UID := $(shell id -u)
export LAB_GID := $(shell id -g)

COMPOSE = docker compose -f deploy/docker-compose.yml --env-file deploy/versions.env \
          $(if $(wildcard .env),--env-file .env) $(if $(wildcard deploy/image-digests.env),--env-file deploy/image-digests.env)
PROMTOOL = $(shell command -v promtool 2>/dev/null || ls .tools/*/promtool 2>/dev/null | head -1)
TS = $$(date -u +%Y%m%dT%H%M%SZ)

.PHONY: help init venv venv-torch dirs weights fixtures reference build up-gpu up-cpu up-both up-monitoring down down-workload \
        controller controller-stop preflight preflight-cpu validate validate-gpu validate-cpu benchmark benchmark-quick \
        freeze-baseline soak overload startup calibrate freeze-tolerances test-gpu fault-test report test rules-test dashboards pin-images lock clean-faults

help:
	@sed -n 's/^## //p' $(MAKEFILE_LIST)

## init            create dirs, .env (random Grafana password), venv with host tooling
init: dirs venv .env
.env:
	@umask 077; printf 'GRAFANA_ADMIN_PASSWORD=%s\n' "$$(head -c 18 /dev/urandom | base64 | tr -dc A-Za-z0-9)" > .env
	@echo "created .env (Grafana admin password, not committed)"
venv:
	@test -x $(PY) || python3 -m venv .venv
	@$(PY) -m pip install -q -r requirements-dev.txt
## venv-torch      (optional, dev machines) CPU torch + torchvision in the host venv for local model tests
venv-torch: venv
	@$(PY) -m pip install -q torch torchvision
dirs:
	@mkdir -p artifacts/{logs,alerts,controller,deploy,faults,preflight,validation,benchmark,soak,overload,startup,review,report} \
	          artifacts/faults/inject/{gpu,cpu,fault} monitoring/targets manifests/weights manifests/reference
	@touch artifacts/.gitkeep monitoring/targets/.gitkeep

## weights         fetch + pin ResNet18 weights (the only download) using the pinned image
weights: dirs
	$(COMPOSE) --profile tools-cpu run --rm --build tools-cpu python -m manifests.fetch_weights
## fixtures        build the 256 fixed inputs + manifest (synthetic unless --source-dir is used)
fixtures: dirs
	$(COMPOSE) --profile tools-cpu run --rm tools-cpu python -m manifests.build_inputs
## reference       CPU FP32 reference logits generated inside the pinned image
reference: dirs
	$(COMPOSE) --profile tools-cpu run --rm tools-cpu python -m manifests.make_reference

build:
	$(COMPOSE) --profile gpu --profile cpu build inference-gpu inference-cpu

## up-gpu          deploy monitoring + GPU workload + DCGM exporter (writes artifacts/deploy/*.log)
up-gpu: dirs .env
	@TS=$(TS); { echo "# deploy log $$TS"; echo "commit: $$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	  echo "lock: $$(sha256sum requirements.lock 2>/dev/null | cut -c1-16 || shasum -a 256 requirements.lock | cut -c1-16)"; \
	  $(PY) deploy/set_targets.py gpu; \
	  $(COMPOSE) --profile gpu up -d --build; \
	  $(COMPOSE) --profile gpu ps; \
	  for s in inference-gpu; do echo "image: $$(docker inspect --format '{{.Image}}' lab-$$s)"; done; \
	} 2>&1 | tee artifacts/deploy/up-gpu-$$TS.log
## up-cpu          deploy monitoring + CPU workload (independent of any GPU)
up-cpu: dirs .env
	@TS=$(TS); { echo "# deploy log $$TS"; echo "commit: $$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	  $(PY) deploy/set_targets.py cpu; $(COMPOSE) --profile cpu up -d --build; $(COMPOSE) --profile cpu ps; \
	} 2>&1 | tee artifacts/deploy/up-cpu-$$TS.log
## up-both         both workload profiles (needed for the CPU-vs-GPU benchmark matrix)
up-both: dirs .env
	$(PY) deploy/set_targets.py both
	$(COMPOSE) --profile gpu --profile cpu up -d --build
up-monitoring: dirs .env
	$(COMPOSE) up -d prometheus alertmanager alert-receiver grafana node-exporter

## down            stop everything (named volumes and ./artifacts are kept)
down: controller-stop
	$(COMPOSE) --profile gpu --profile cpu --profile fault --profile tools-gpu --profile tools-cpu down
## down-workload   stop only workload containers; monitoring keeps running
down-workload:
	$(COMPOSE) --profile gpu --profile cpu --profile fault rm -sf inference-gpu inference-cpu inference-fault

## controller      start the host recovery controller in the background (needs the Docker socket)
controller: dirs
	@test -f artifacts/controller/controller.pid && kill -0 $$(cat artifacts/controller/controller.pid) 2>/dev/null && echo "controller already running" || \
	  { nohup $(PY) -m ops.controller --config ops/controller.yaml > artifacts/controller/controller.out 2>&1 & echo $$! > artifacts/controller/controller.pid; echo "controller started (pid $$(cat artifacts/controller/controller.pid))"; }
controller-stop:
	@test -f artifacts/controller/controller.pid && kill $$(cat artifacts/controller/controller.pid) 2>/dev/null || true
	@rm -f artifacts/controller/controller.pid

## preflight       inventory + PF01-PF06 -> artifacts/preflight/<run>/preflight.json (PROFILE=gpu|cpu)
preflight: dirs
	$(PY) -m preflight.run_preflight --mode $(PROFILE) --image $(IMAGE)
preflight-cpu: dirs
	$(PY) -m preflight.run_preflight --mode cpu --image $(IMAGE)

## validate        correctness (in container) + API contract checks for the selected PROFILE
validate: validate-$(PROFILE)
validate-gpu:
	$(COMPOSE) --profile tools-gpu run --rm tools-gpu python -m validation.correctness --backend cuda
	$(PY) -m validation.api_checks --url http://127.0.0.1:8000 --expect-backend cuda
validate-cpu:
	$(COMPOSE) --profile tools-cpu run --rm tools-cpu python -m validation.correctness --backend cpu
	$(PY) -m validation.api_checks --url http://127.0.0.1:8001 --expect-backend cpu

## calibrate      measure CPU-vs-GPU error on the calibration subset (informational; does not freeze)
calibrate:
	$(COMPOSE) --profile tools-gpu run --rm tools-gpu python -m validation.correctness --backend cuda --calibrate
## freeze-tolerances  freeze the reviewed tolerances (commit validation/tolerances.json afterwards)
freeze-tolerances:
	$(COMPOSE) --profile tools-gpu run --rm tools-gpu python -m validation.correctness --freeze
## test-gpu        GPU integration tests inside the GPU tools container (real hardware evidence)
test-gpu:
	$(COMPOSE) --profile tools-gpu run --rm tools-gpu python -m pytest -m gpu -p no:cacheprovider

## benchmark       18-cell x 3-trial baseline (needs `make up-both`); resumable
benchmark:
	$(PY) -m benchmarks.run_matrix --config benchmarks/matrix.yaml $(if $(EXPERIMENT),--experiment-id $(EXPERIMENT) --resume)
benchmark-quick:  # NOT acceptance evidence: shortened runs to check the plumbing
	$(PY) -m benchmarks.run_matrix --config benchmarks/matrix.yaml --duration 5 --trials 1 --experiment-id quick-$$(date -u +%H%M%S)
## freeze-baseline freeze the matching p95 baseline for LatencyRegression (BASELINE=<summary.json>)
freeze-baseline:
	$(PY) -m benchmarks.freeze_baseline --summary $(BASELINE) --concurrency 4
	curl -fsS -X POST http://127.0.0.1:9090/-/reload && echo "prometheus reloaded"
soak:
	$(PY) -m benchmarks.soak --url $(if $(filter cpu,$(PROFILE)),http://127.0.0.1:8001 --backend cpu,http://127.0.0.1:8000 --backend cuda)
overload:
	$(PY) -m benchmarks.overload --url $(if $(filter cpu,$(PROFILE)),http://127.0.0.1:8001 --backend cpu,http://127.0.0.1:8000 --backend cuda)
startup:
	$(PY) -m benchmarks.measure_startup --profile $(PROFILE)

## fault-test      5 scenarios x 3 trials -> artifacts/faults/incidents.jsonl (needs controller + monitoring up)
fault-test:
	$(PY) -m faults.run_faults --profile $(PROFILE) $(FAULT_ARGS)
clean-faults:
	rm -f artifacts/faults/inject/*/delay_ms
	-$(COMPOSE) --profile fault rm -sf inference-fault
	$(PY) deploy/set_targets.py remove-fault

## report          summary.json + REPORT.md + evidence index from raw artifacts
report:
	$(PY) -m report.generate_report $(if $(EXPERIMENT),--experiment $(EXPERIMENT))

## test            unit tests (CPU; fixtures labelled simulated where used)
test:
	$(PY) -m pytest -m "not gpu and not integration"
## rules-test      promtool check + alert rule tests (local promtool or the pinned Prometheus image)
rules-test:
	@if [ -n "$(PROMTOOL)" ]; then $(PROMTOOL) check rules monitoring/rules/alerts.yml monitoring/rules/baseline.yml && $(PROMTOOL) test rules monitoring/tests/alerts_test.yml; \
	else . deploy/versions.env && docker run --rm --entrypoint promtool -v "$$PWD/monitoring:/m:ro" $$PROMETHEUS_IMAGE test rules /m/tests/alerts_test.yml; fi
dashboards:
	$(PY) monitoring/grafana/build_dashboards.py

## pin-images      resolve image tags to immutable digests -> deploy/image-digests.env
pin-images:
	@. deploy/versions.env; : > deploy/image-digests.env; \
	for v in BASE_IMAGE PROMETHEUS_IMAGE ALERTMANAGER_IMAGE GRAFANA_IMAGE NODE_EXPORTER_IMAGE DCGM_EXPORTER_IMAGE; do \
	  ref=$${!v}; docker pull -q $$ref >/dev/null; d=$$(docker inspect --format '{{index .RepoDigests 0}}' $$ref); echo "$$v=$$d" >> deploy/image-digests.env; echo "$$v=$$d"; done
## lock            regenerate requirements.lock from the host venv
lock:
	$(PY) -m pip freeze > requirements.lock
