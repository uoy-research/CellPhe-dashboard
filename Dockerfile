FROM python:3.12-slim AS builder

# Build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-compile --no-cache-dir --user -r requirements.txt

# Copy application code
COPY CellPheDashboard.py .

# Application image
FROM python:3.12-slim AS app
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/CellPheDashboard.py /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["python", "-m", "streamlit", "run", "CellPheDashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
