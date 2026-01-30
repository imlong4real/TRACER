# syntax=docker/dockerfile:1

FROM python:3.10-slim

ARG TORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG TORCHAUDIO_VERSION=2.1.2
ARG TORCH_CPU_URL=https://download.pytorch.org/whl/cpu
ARG PYG_WHL_URL=https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        g++ \
        pkg-config \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        libspatialindex-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel Cython \
    && python -m pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url ${TORCH_CPU_URL} \
    && python -m pip install torch-geometric --find-links ${PYG_WHL_URL} \
    && python -m pip install -e .

CMD ["python", "-c", "import hotnerd; print(hotnerd.__version__)" ]
