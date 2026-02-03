# Use CUDA runtime base image (smaller than devel by ~2-3GB)
# cuda-nvcc and cuda-cudart-dev added below for flash-attn source build fallback
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
   PIP_PREFER_BINARY=1 \
   PIP_BREAK_SYSTEM_PACKAGES=1 \
   PYTHONUNBUFFERED=1 \
   CMAKE_BUILD_PARALLEL_LEVEL=8

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
   python3 python3-pip python3-venv curl git git-lfs wget libgl1 libglib2.0-0 \
   python3-dev build-essential gcc \
   cuda-nvcc-12-8 cuda-cudart-dev-12-8 \
   && ln -sf /usr/bin/python3 /usr/bin/python \
   && ln -sf /usr/bin/pip3 /usr/bin/pip \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir jupyterlab \
    jupyter-server jupyter-server-terminals \
    ipykernel huggingface_hub[cli] \
    ninja packaging

# Create the final image
FROM base AS final

# Shallow clone diffusion-pipe and strip .git to save image space
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/tdrussell/diffusion-pipe /diffusion_pipe && \
    rm -rf /diffusion_pipe/.git /diffusion_pipe/**/.git

# Install requirements but exclude flash-attn to avoid build issues
RUN grep -v -i "flash-attn\|flash-attention" /diffusion_pipe/requirements.txt > /tmp/requirements_no_flash.txt && \
    pip install --no-cache-dir -r /tmp/requirements_no_flash.txt && \
    pip install --no-cache-dir --upgrade deepspeed

# Clean up build artifacts
RUN rm -rf /tmp/* /root/.cache

COPY src/start_script.sh /start_script.sh
RUN chmod +x /start_script.sh
CMD ["/start_script.sh"]
