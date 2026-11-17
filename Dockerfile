FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image ~800MB instead of ~6GB. For GPU serving,
# swap the base for an nvidia/cuda image and drop the index-url flag.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/

# THRESHOLD and INPUT_SIZE are deliberately not set: the service takes both from the checkpoint.
ENV MODEL_PATH=artifacts/patchcore_carpet.pt \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
