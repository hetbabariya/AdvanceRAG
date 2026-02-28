FROM python:3.11-slim

WORKDIR /app

# Copy entire project
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Make sure Python sees project root
ENV PYTHONPATH=/app

EXPOSE 10000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "backend.api.main:app", "--bind", "0.0.0.0:10000"]