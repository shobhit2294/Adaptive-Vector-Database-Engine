FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    libgtest-dev \
    nlohmann-json3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j"$(nproc)" && \
    ctest --test-dir build --output-on-failure

FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/build/vectordb_server /usr/local/bin/vectordb_server
COPY --from=builder /app/build/vectordb_bench /usr/local/bin/vectordb_bench
COPY --from=builder /app/build/test_* /app/tests/

RUN mkdir -p /data/vectordb
EXPOSE 8080

CMD ["/usr/local/bin/vectordb_server"]
