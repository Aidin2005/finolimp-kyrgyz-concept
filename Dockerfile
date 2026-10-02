FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    LOKY_MAX_CPU_COUNT=2 \
    PORT=5050

WORKDIR /app

# Install system dependencies needed for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and default datasets (output files already included)
COPY . .

EXPOSE 5050

# Run with gunicorn using dynamic PORT from cloud provider (Render/Railway)
CMD exec gunicorn --bind 0.0.0.0:${PORT:-5050} --workers 2 --timeout 180 app:app
