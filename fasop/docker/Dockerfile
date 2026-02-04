#
# FASOP ProfilerDB Docker Image
# For FX graph node-level profiling to generate ProfileDB
#
# Build: docker build -t make_profiledb:latest .
#
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

LABEL maintainer="ETRI-High Performance Computing System Research Section"
LABEL description="FASOP ProfileDB generation environment"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ENV TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=18000

# Install Python dependencies
RUN pip install --no-cache-dir \
    "transformers==4.46.2" \
    datasets \
    accelerate \
    sentencepiece \
    protobuf \
    gpustat \
    packaging

# Set working directory
WORKDIR /workspace/fasop

# Default command
CMD ["/bin/bash"]
