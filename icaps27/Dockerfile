# CPU image for the quick check and for regenerating figures from shipped
# results. GPU runs use the same uv.lock on the host (make env).
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_PYTHON_INSTALL_DIR=/opt/python
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git make && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /usr/local/bin/uv
WORKDIR /artifact
COPY pyproject.toml uv.lock README.md ./
# The lock's exact versions, with CPU-only torch instead of the CUDA wheels.
RUN uv venv --python 3.12 .venv \
    && uv export --frozen --extra dev --no-hashes --no-emit-project \
       | grep -v -E '^(nvidia-|triton|cuda-)' > /tmp/requirements.txt \
    && uv pip install --python .venv/bin/python --torch-backend cpu --no-deps \
       -r /tmp/requirements.txt
COPY . .
RUN uv pip install --python .venv/bin/python --no-deps -e .
ENV PATH=/artifact/.venv/bin:$PATH
CMD ["make", "smoke"]
