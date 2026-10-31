FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# CPU wheel first, so the package install below doesn't pull the CUDA build
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY config ./config
ENV PROPMEASURE_GATE_CONFIG=/app/config/release_gate.yaml

CMD ["uvicorn", "propmeasure.api:app", "--host", "0.0.0.0", "--port", "8000"]
