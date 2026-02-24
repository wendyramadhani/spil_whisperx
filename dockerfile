# 1. Gunakan Base Image resmi NVIDIA (Sudah ada CUDA & Python)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 2. Set Environment Variables 
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 3. Install System Dependencies (FFmpeg, Python, Git)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Buat Alias 'python' ke 'python3'
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 5. Set Folder Kerja
WORKDIR /app

# 6. Install Python Libraries (Sesuai urutan sukses kita tadi)
#    a. Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

#    b. Install PyTorch GPU (Versi Stabil 2.1.2 + CUDA 11.8)
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

#    c. Install WhisperX & Server Tools
RUN pip install --no-cache-dir \
    whisperx \
    fastapi \
    "uvicorn[standard]" \
    python-dotenv \
    python-multipart

COPY . .

# 8. Buka Port 8000
EXPOSE 8000

# 9. Perintah Menjalankan Server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]