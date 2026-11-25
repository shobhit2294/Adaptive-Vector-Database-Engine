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

FROM ubuntu:24.04 AS test-runner
COPY --from=builder /app/build /app/build
WORKDIR /app
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
