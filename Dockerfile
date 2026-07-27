FROM python:3.11-slim

# Install necessary system dependencies for aiortc/av and opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend source code
COPY . .

# Default command (this can be overridden in Render dashboard for different workers)
# e.g. command: python3 navtalk_worker.py OR python3 agent.py start
CMD ["python3", "navtalk_worker.py"]
