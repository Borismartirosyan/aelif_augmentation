# Use the official PyTorch Docker image as the base image for latest torch
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

ENV PYTHONPATH="/workspace:/workspace/diffusers:/workspace/CLIP:/workspace/aelif/:/workspace/aelif/aelif_augmentation_inference/:/workspace/aelif:{PYTHONPATH}"
# Install additional system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ca-certificates \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    omegaconf==2.3.0 \
    transformers==4.47.1 \
    pydantic==2.10.4 \
    bitsandbytes==0.45.0 \
    pillow==11.1.0 \
    accelerate==1.2.1 \
    diffusers==0.32.1 \
    protobuf==5.29.3 \
    sentencepiece==0.2.0

# Authenticate Hugging Face CLI
RUN huggingface-cli login --token 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

# Set working directory
WORKDIR /workspace

RUN mkdir -p /root/.cache/huggingface/

# Clone and install diffusers
# RUN git clone https://github.com/Borismartirosyan/diffusers.git \
#     && cd diffusers \
#     && pip install -e .

# Clone and install CLIP
RUN git clone https://github.com/openai/CLIP.git \
    && cd CLIP \
    && pip install -e .

# Copy main.py into the container
COPY . /workspace/aelif

# Set working directory
WORKDIR /workspace/aelif

RUN mkdir res
# RUN /opt/conda/bin/python main.py
# # Run main.py as default
CMD ["python3", "main.py"]