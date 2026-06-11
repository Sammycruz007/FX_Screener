# Use a lightweight, stable Python base image
FROM python:3.11-slim

# Set system environment variables to optimize Python inside Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the functional workspace inside the container
WORKDIR /app

# Install dependencies first (enables Docker build caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and entry script
COPY src/ ./src/
COPY run_pipeline.py .

# Define the execution entrypoint when the container runs
CMD ["python", "run_pipeline.py"]