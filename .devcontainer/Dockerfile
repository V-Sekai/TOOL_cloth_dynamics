FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Disable git pager to prevent interactive prompts
ENV GIT_PAGER=cat

# Install build essentials and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspaces/TOOL_cloth_dynamics

# Keep container running
CMD ["/bin/bash"]
