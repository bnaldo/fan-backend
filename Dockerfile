FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8001

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
