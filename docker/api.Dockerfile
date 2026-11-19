FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch wheels (~200 MB instead of several GB of CUDA libraries).
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/models /app/data/feedback \
    && chown -R app:app /app
USER app

ENV MODEL_PATH=/app/models/avocado_ripeness_resnet50.pth \
    FEEDBACK_PATH=/app/data/feedback/feedback.csv

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "avocado_ai.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
