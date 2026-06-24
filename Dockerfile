# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
    
WORKDIR /app

COPY . /app

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
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install -e .

CMD ["python", "-c", "import tracer; print(tracer.__version__)" ]
