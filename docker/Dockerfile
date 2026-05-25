FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gfortran \
      git \
      python3 \
      python3-pip \
      python3-venv \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /workspace
RUN make core

CMD ["python3", "scripts/test_matrix.py", "demo", "--trace-index", "3"]
