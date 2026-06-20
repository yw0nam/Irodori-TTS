FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg libsndfile1 curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
# uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app
COPY . /app
# uv.lock 기반 의존성 설치 (git deps: dacvae, silentcipher 포함)
RUN uv sync --frozen --no-dev || uv sync --no-dev
ENV HF_HOME=/hf
EXPOSE 8091
ENTRYPOINT ["uv","run","python","api_server.py"]
CMD ["--checkpoint","Aratako/Irodori-TTS-500M-v3","--devices","cuda:0","--precision","bf16","--num-worker","2","--port","8091"]
