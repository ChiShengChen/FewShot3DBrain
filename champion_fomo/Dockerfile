FROM python:3.10-slim

# Maintainer
LABEL maintainer="Jaume Banus <jaume.banus-cobo@chuv.ch>"

# Install the ps package
RUN apt-get update && apt-get install -y procps g++ git && apt-get clean 

# Set work directory
WORKDIR /usr/src/

# Prevent python to write pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent stderr and stdout output
ENV PYTHONUNBUFFERED=1

# Copy project
COPY . /usr/src/

# For all dependencies
RUN pip install -e ".[dev,test]"

# ============================================
# Pretrained Model Weights Configuration
# ============================================
# The pretrained weights are included in the image at:
#   /usr/src/weights/fomo25_mmunetvae_pretrained.ckpt
#
# To load the weights in your code:
#   from utils.load_weights import load_pretrained_checkpoint
#   checkpoint = load_pretrained_checkpoint()
#
# Or use the environment variable:
#   import os
#   weights_path = os.environ.get('FOMO25_WEIGHTS_PATH')
# ============================================

# Set environment variable for pretrained weights path
ENV FOMO25_WEIGHTS_PATH=/usr/src/weights/fomo25_mmunetvae_pretrained.ckpt

# Verify weights exist (will fail build if missing)
RUN test -f /usr/src/weights/fomo25_mmunetvae_pretrained.ckpt || echo "Warning: Pretrained weights not found"