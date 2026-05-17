FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential libspatialindex-dev \
    libgeos-dev libproj-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Predownload DINOv2 weights into the image (optional, comment out for faster build)
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitg14')" || true

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
