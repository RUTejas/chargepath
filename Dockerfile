FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .

# Directories
RUN mkdir -p checkpoints results/figures data/weather_cache static/icons

EXPOSE 5000
ENV PORT=5000

CMD ["python", "app.py"]
