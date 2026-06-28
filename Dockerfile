# Single image for both the API and the ingestion worker.
# Default command runs the API; the worker task overrides CMD with:
#   python -m app.worker
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /code

# install deps first (layer caching — deps change rarely)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the application package and scripts
COPY app/ ./app/
COPY scripts/ ./scripts/

# API listens on 8000
EXPOSE 8000

# default: run the API. uvicorn binds 0.0.0.0 so it's reachable from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
