# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by TensorFlow
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project
COPY . .

# Expose Cloud Run port
EXPOSE 8080

# Run Flask with Gunicorn (Cloud Run expects 8080)
CMD ["gunicorn", "-b", ":8080", "app:app"]
