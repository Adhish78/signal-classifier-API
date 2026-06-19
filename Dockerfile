# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production runtime environment
FROM python:3.12-slim AS runner

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /root/.local /root/.local
COPY api/ /app/api/
COPY src/ /app/src/


# Expose local user bin directory where uvicorn is installed
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
