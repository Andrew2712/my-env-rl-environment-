# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# ── Working directory ───────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies first (better caching) ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application files ──────────────────────────────────────────────────────
COPY inference.py .
COPY models.py .
COPY client.py .
COPY openenv.yaml .
COPY README.md .

# Copy the entire server directory (important: trailing slash)
COPY server/ ./server/

# ── Expose port (required for HF Spaces) ────────────────────────────────────────
EXPOSE 7860

# ── Health check (improved) ─────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=5 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health', timeout=5).raise_for_status()" || exit 1

# ── Run the app ─────────────────────────────────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
