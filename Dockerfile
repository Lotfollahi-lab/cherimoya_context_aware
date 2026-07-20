# droast ignore=DF007 reason="added a .dockerignore file"
FROM python:3.12@sha256:1a7dbae78e9568b95a8e2934719c4bf984d7ae319b3659ed129f59d8aca6d7a8 AS builder

WORKDIR /work
RUN cat /etc/os-release
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
COPY . .
# droast ignore=DF051 reason="pip should be latest"
RUN python -m pip install --no-cache-dir --upgrade pip
# droast ignore=DF051 reason="cherimoya installation is from pwd"
RUN pip install --no-cache-dir . 

FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS main

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["python3", "-c", "from cherimoya import Cherimoya; print('cherimoya imported OK')"]

