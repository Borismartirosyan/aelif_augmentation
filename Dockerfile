# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install git and any required dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy all necessary files into the container
COPY . /app

# Clone and install the diffusers repository
RUN git clone https://github.com/Borismartirosyan/diffusers.git \
    && cd diffusers \
    && pip install -e .

# Login to Hugging Face
RUN huggingface-cli login --token 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

RUN pip install -U bitsandbytes

RUN cd /content/aelif_augmentation/ && mkdir res

# Clone and install the CLIP repository
RUN git clone https://github.com/openai/CLIP.git \
    && cd CLIP \
    && pip install -e .

RUN pip3 install -r requirements.txt

# Command to run the container
CMD ["cd", "/content/aelif_augmentation/", "&&", "python3", "main.py"]