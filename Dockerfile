# Start from an official slim Python image
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies first (layer caching: deps change rarely)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker code in
COPY worker.py .

# Run the worker when the container starts
CMD ["python", "worker.py"]