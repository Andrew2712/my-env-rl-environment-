# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment ─────────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# ── Working directory ───────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ────────────────────────────────────────────────────────
# requirements.txt is at project root — copy it first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project ─────────────────────────────────────────────────────────────────
# Copy models.py and client.py (project root level)
COPY models.py    .
COPY client.py    .
COPY openenv.yaml .
COPY README.md    .
COPY inference.py .

# Copy server package
COPY server/      ./server/

# ── Expose port (HF Spaces requires 7860) ───────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c \
  "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

# ── Launch ────────────────────────────────────────────────────────────────────────
# --workers 1: environment has shared singleton registry state.
# Multiple workers would split the registry across processes.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
