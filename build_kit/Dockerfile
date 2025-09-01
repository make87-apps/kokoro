# ---------- Build ----------
FROM rust:1.86-slim-bookworm AS build
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl pkg-config build-essential \
    libssl-dev && \
    rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --release

# ---------- Runtime ----------
FROM debian:bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# App
COPY --from=build /app/target/release/app /usr/local/bin/app

# ONNX Runtime (CPU)
ARG ORT_VERSION=1.22.0
RUN mkdir -p /opt/onnxruntime && \
    curl -L "https://sourceforge.net/projects/onnx-runtime.mirror/files/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz/download" \
      -o /tmp/ort.tgz && \
    tar -xzf /tmp/ort.tgz -C /opt && \
    mv /opt/onnxruntime-linux-x64-${ORT_VERSION} /opt/onnxruntime && \
    rm -f /tmp/ort.tgz
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
ENV ORT_THREADS=2

# Kokoro model + tokenizer + voice
RUN mkdir -p /models/onnx /models/voices
RUN curl -L "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_quantized.onnx?download=true" \
      -o /models/onnx/model_quantized.onnx && \
    curl -L "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/tokenizer.json?download=true" \
      -o /models/tokenizer.json && \
    curl -L "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af_bella.bin?download=true" \
      -o /models/voices/af_bella.bin

ENV KOKORO_MODEL=/models/onnx/model_quantized.onnx
ENV KOKORO_TOKENIZER=/models/tokenizer.json
ENV KOKORO_VOICE=/models/voices/af_bella.bin
ENV KOKORO_SPEED=1.0

CMD ["/usr/local/bin/app"]
