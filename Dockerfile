# Recommendation server (stdlib only, no pip install needed).
# Secrets are NOT baked in: they come from the environment at run time
# (docker compose reads .env; .dockerignore keeps .env out of the image).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

WORKDIR /app
COPY src ./src
COPY data ./data

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-B", "-m", "food_crag.server"]
