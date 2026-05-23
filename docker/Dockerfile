# Stage 1: Build stage
FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install build pre-requisites
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    cmake \
    autoconf \
    libomp-dev \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /openfhe-development

# Copy the current directory contents into the container
COPY . .

# Create build directory and build (using BuildKit cache mounts for faster rebuilds)
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/openfhe-development/build/_deps \
    mkdir -p build && cd build && \
    cmake .. -DBUILD_UNITTESTS=OFF -DBUILD_BENCHMARKS=OFF -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache && \
    make -j$(nproc) advanced-ckks-bootstrapping gelu-piecewise-cpu layernorm-cpu softmax-cpu

# Stage 2: Runtime stage
FROM ubuntu:22.04 AS runner

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install runtime dependencies (OpenMP runtime libraries)
RUN apt-get update && apt-get install -y \
    libomp5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /openfhe-development

# Copy shared libraries, binaries, and demo data from the builder stage
COPY --from=builder /openfhe-development/build/lib/ /usr/local/lib/
COPY --from=builder /openfhe-development/build/bin/ /openfhe-development/build/bin/
COPY --from=builder /openfhe-development/demoData/ /openfhe-development/demoData/

# Update the dynamic linker cache to find OpenFHE shared libraries
RUN ldconfig

# Run the default example
CMD ["/openfhe-development/build/bin/examples/pke/softmax-cpu"]
