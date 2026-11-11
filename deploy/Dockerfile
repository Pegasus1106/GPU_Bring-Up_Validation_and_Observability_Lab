# Workload image: inference service, preflight, validation, benchmark tooling.
# The same image serves the CPU and GPU profiles so baselines compare like with like.
# Weights and input fixtures are NOT baked in: they are mounted read-only from ./manifests
# after `make weights`, so the service never downloads anything at start-up.
ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /opt/lab

COPY requirements-app.txt requirements-test.txt ./
RUN pip install -r requirements-app.txt -r requirements-test.txt

COPY pyproject.toml ./
COPY app ./app
COPY common ./common
COPY preflight ./preflight
COPY validation ./validation
COPY loadgen ./loadgen
COPY benchmarks ./benchmarks
COPY ops ./ops
COPY faults ./faults
COPY report ./report
COPY tests ./tests

# Run as an unprivileged user (GPU access does not require root inside the container).
RUN useradd --create-home --uid 10001 lab && mkdir -p /artifacts && chown lab /artifacts
USER lab

EXPOSE 8000
CMD ["python", "-m", "app.main"]
