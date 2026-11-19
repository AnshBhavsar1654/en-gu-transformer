# syntax=docker/dockerfile:1
#
# Targets:
#   api        inference API; the model is mounted at /models/best.pt   (docker compose uses this)
#   api-baked  same, with models/best.pt copied into the image           (for hosts without volumes, e.g. Cloud Run)
#   ui         Streamlit front end, a thin client of the API (no torch)
#
#   python -m en_gu fetch-model                     # writes models/best.pt (+ version sidecar) from the registry
#   docker build --target api --build-arg GIT_SHA=$(git rev-parse HEAD) -t en-gu-api .
#   docker run -p 8000:8000 -v "$PWD/models:/models:ro" en-gu-api

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------- builder: venv with CPU-only torch
FROM python:${PYTHON_VERSION}-slim AS builder
ARG TORCH_VERSION=2.12.1
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# The CPU wheel is ~200 MB instead of ~2 GB with CUDA; installed first so this layer is cached across code changes
RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu \
    # C++ headers and test binaries are only needed to build extensions (~150 MB)
    && rm -rf /opt/venv/lib/python*/site-packages/torch/include /opt/venv/lib/python*/site-packages/torch/test
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

# ---------------------------------------------------------------------------- api: model mounted at runtime
FROM python:${PYTHON_VERSION}-slim AS api
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /models /data \
    && chown app:app /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY configs ./configs
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    EN_GU_MODEL_SOURCE=path \
    EN_GU_MODEL_PATH=/models/best.pt \
    EN_GU_FEEDBACK_DB=/data/feedback.db
USER app
EXPOSE 8000
# Liveness only; readiness (model loaded) is GET /ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
# The git commit, set by CI (--build-arg GIT_SHA=...); GET /info reports it so a deploy can be verified.
# Declared last so a new commit doesn't invalidate the cached layers above.
ARG GIT_SHA=
LABEL org.opencontainers.image.source="https://github.com/AnshBhavsar1654/en-gu-transformer" \
      org.opencontainers.image.revision="${GIT_SHA}"
ENV EN_GU_BUILD_SHA=${GIT_SHA}
CMD ["python", "-m", "en_gu", "serve", "--host", "0.0.0.0"]

# ---------------------------------------------------------------------------- api-baked: model inside the image
FROM api AS api-baked
COPY --chown=app:app models/best.pt* /models/

# ---------------------------------------------------------------------------- ui: Streamlit client
FROM python:${PYTHON_VERSION}-slim AS ui
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN pip install "streamlit>=1.40.0" "httpx>=0.27" \
    && useradd --create-home --uid 10001 app
WORKDIR /app
# In API mode the app needs nothing from the package (STANDALONE=1 is not supported in this image: no torch)
COPY app.py ./
ENV EN_GU_API_URL=http://api:8000
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
