# True ASI System - Production Docker Container
# Based on NVIDIA PyTorch container for optimal AI/ML performance

FROM python:3.11-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional AI/ML packages that might be missing
RUN pip install --no-cache-dir \
    accelerate \
    datasets \
    tokenizers \
    safetensors \
    sentencepiece \
    protobuf

# Copy the entire codebase
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/models /app/data

# Set permissions
RUN chmod +x /app/*.py

# Expose ports for web interfaces and APIs
EXPOSE 8000 8080 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('PyTorch version:', torch.__version__)" || exit 1

# Default command - run the main ASI system
CMD ["python", "MASTER_ASI_INTEGRATION_2025.py"]
