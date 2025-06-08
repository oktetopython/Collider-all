# Base image: NVIDIA CUDA
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Maintainer label
LABEL maintainer="your_email@example.com"

# Set environment variables to non-interactive (prevents prompts during build)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential build tools and dependencies
RUN apt-get update && \
    apt-get install -y \
    cmake \
    git \
    build-essential \
    libboost-all-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app/cuda-bsgs

# Copy the application source code into the container
COPY . /app/cuda-bsgs

# Create a build directory and build the project
RUN cmake -S . -B build -G Ninja && \
    cmake --build build --parallel $(nproc)

# Default command to run the application (example)
# Users should override this or use the compiled binary directly
CMD ["./build/bsgs", "--help"]

# Expose any necessary ports (if applicable, not typical for this kind of app)
# EXPOSE 8080

# Set user (optional, good practice for security)
# USER appuser
```
