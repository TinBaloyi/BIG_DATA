# Dockerfile.cpu
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget ca-certificates libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace
ENV PYTHONUNBUFFERED=1
CMD ["/bin/bash"]
