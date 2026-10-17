FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e . && \
    python -m castled.core._build_engine && \
    rm -f castled/core/_engine.c castled/core/_engine.o castled/core/src/*.o

ENV LICHESS_TOKEN=""
CMD ["python", "tools/lichess_bot.py", "--config", "tools/lichess-bot-config.yml"]

