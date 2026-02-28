FROM python:3.11-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy entire project
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Make sure Python sees project root
ENV PYTHONPATH=/app

# Render provides PORT automatically
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker backend.api.main:app --bind 0.0.0.0:$PORT"]