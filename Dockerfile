# RunPod Serverless worker for nvidia/LocateAnything-3B
# https://huggingface.co/nvidia/LocateAnything-3B
#
# Model weights are NOT baked into this image. We rely on RunPod's
# "cached models" feature instead (set in the endpoint's Model field),
# which is faster and cheaper than downloading ~8GB on every cold start.
# See handler.py for how the cached snapshot is located.
 
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
 
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
 
# Ubuntu 22.04 ships Python 3.10 by default.
# ffmpeg/libgl/libglib are needed by opencv-python-headless and decord,
# which nvidia/LocateAnything-3B's custom modeling code imports.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /
 
# Install dependencies first so Docker can cache this layer between builds
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt
 
# Worker code
COPY src/handler.py /handler.py
 
CMD ["python3", "-u", "/handler.py"]
 
