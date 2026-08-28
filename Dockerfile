# Build — generate certs/{cert.pem,key.pem} FIRST (see CLAUDE.md); they're baked into
# the image (COPY . . below, not excluded by .dockerignore) so gunicorn can terminate
# TLS without a runtime bind-mount. Rebuild the image if you regenerate the cert.
# Never push/share this image publicly — it contains the TLS private key.
#
# Run — pass all required env vars at runtime (never bake secrets into the image)

#  docker run -p 127.0.0.1:8080:8080 \
#    -e ANTHROPIC_API_KEY=sk-... \
#    -e OPENAI_API_KEY=sk-... \
#    -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#    -e OLLAMA_API_KEY=... \
#    -e DB_HOST=host.docker.internal \
#    -e DB_NAME=yourdb \
#    -e DB_USER=youruser \
#    -e DB_PASSWORD=yourpassword \
#    -e LOG_LEVEL=INFO \
#    -v $(pwd)/logs:/app/logs \
#    pubmed-ai
#
#  A few notes:
#  - -p 127.0.0.1:8080:8080 — binds the published port to the host loopback only; other machines on your LAN cannot reach it.
#    gunicorn must bind to 0.0.0.0 inside the container (see CMD below) for Docker's forwarding to work — this is safe
#    because the container has its own isolated network namespace.
#  - Serving — gunicorn (wsgi:application), not the Flask dev server. `python app.py` (Werkzeug's dev server, with its
#    interactive debugger) is for local development only — see CLAUDE.md.
#  - libgomp1 — the only system dep needed; onnxruntime (fastembed's backend) requires it at runtime on Debian slim images
#  - Model pre-baked — BAAI/bge-small-en-v1.5 is downloaded during docker build so the container starts immediately without a model download delay
#  - -v $(pwd)/logs:/app/logs — mounts the log directory as a volume so logs survive container restarts
#  - DB_HOST=host.docker.internal — use this when your Postgres is running on the host machine; replace with the actual hostname/IP if Postgres is elsewhere
#  - OPENAI_API_KEY / OLLAMA_BASE_URL / OLLAMA_API_KEY — all optional; omit any to leave that provider out of the
#    chat model dropdown. Local Ollama runs on the host, so it needs host.docker.internal the same way DB_HOST does
#    (Ollama Cloud, via OLLAMA_API_KEY, is a public endpoint and needs no such mapping).
#  - host.docker.internal resolves automatically on Docker Desktop (Mac/Windows). On native Linux Docker Engine,
#    add --add-host=host.docker.internal:host-gateway to the `docker run` command above or it won't resolve.
#


FROM python:3.12-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp is required by onnxruntime (used by fastembed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download embedding model ──────────────────────────────────────────────
# Bakes BAAI/bge-small-en-v1.5 into the image so startup is instant
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
    && mkdir -p logs

# ── Application code ──────────────────────────────────────────────────────────
COPY . .


EXPOSE 8080

# --timeout 300 — gunicorn's default (30s) kills+respawns a worker (SIGKILL, logged as
# "WORKER TIMEOUT" / "Perhaps out of memory?" — that message is boilerplate, not a real
# OOM diagnosis) if it goes quiet mid-request. Saving a large collection (up to
# MAX_RESULTS_MAX=200 articles: PMC full-text fetches + chunk embedding, all synchronous
# in one request) routinely exceeds 30s.
# --certfile/--keyfile — terminate TLS in gunicorn itself, using certs/{cert.pem,key.pem}
# baked into the image by COPY . . above (generate them per CLAUDE.md BEFORE running
# docker build, or the worker fails to boot with a missing-file error). Required:
# COOKIE_SECURE defaults to true (auth cookies marked Secure), which browsers silently
# refuse to store/send over a plain-HTTP connection.
CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "4", "--timeout", "300", "--certfile", "/app/certs/cert.pem", "--keyfile", "/app/certs/key.pem", "wsgi:application"]
