# Build llama-server for native ARM64 support (Apple Silicon)
FROM debian:bookworm-slim AS build

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp
WORKDIR /build
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

WORKDIR /build/llama.cpp
RUN cmake -B build \
    -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=ON \
    && cmake --build build --config Release --target llama-server -j$(nproc)

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libcurl4 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the server binary and all built shared libraries
COPY --from=build /build/llama.cpp/build/bin/llama-server /app/llama-server
RUN mkdir -p /app/lib
COPY --from=build /build/llama.cpp/build/ /tmp/build/
RUN find /tmp/build -name "*.so*" -exec cp {} /app/lib/ \; && rm -rf /tmp/build

WORKDIR /app
ENV LD_LIBRARY_PATH=/app/lib
ENTRYPOINT ["/app/llama-server"]

