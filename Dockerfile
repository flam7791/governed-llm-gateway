# The gateway as a container.
# Build:  docker build -t governed-llm-gateway .
# Run:    docker run --rm -p 127.0.0.1:8080:8080 \
#           -v "$PWD/gateway.json:/config/gateway.json:ro" -v llmgw-data:/data \
#           -e LLMGW_KEY_RESEARCH=... -e ANTHROPIC_API_KEY=... governed-llm-gateway
# The configuration is mounted read-only; the ledger lives on the /data volume; secrets come in
# as environment variables (from a secret store in a real deployment), never in the image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLMGW_DB_PATH=/data/gateway.db

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[azure]"

RUN useradd --create-home --uid 10001 appuser && mkdir -p /data /config \
    && chown appuser /data
USER appuser
VOLUME ["/data"]

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"
CMD ["llmgw", "serve", "--config", "/config/gateway.json", "--host", "0.0.0.0", "--port", "8080"]
