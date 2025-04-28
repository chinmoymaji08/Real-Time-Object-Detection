FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ByteTrack
RUN git clone https://github.com/ifzhang/ByteTrack.git && \
    cd ByteTrack && \
    pip install -e .

# Install cython-bbox package (replacement for cython_bbox)
RUN pip install cython-bbox

# Copy your application code
COPY . .

# Set default command
ENTRYPOINT ["python", "main.py"]

