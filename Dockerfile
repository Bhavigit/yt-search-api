# Use the official lightweight Python image as the base
FROM python:3.10-slim

# Set environment variables:
# - PYTHONDONTWRITEBYTECODE: prevents Python from writing .pyc files
# - PYTHONUNBUFFERED: ensures real-time output in logs (stdout/stderr)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container to /app
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \  # Needed to compile some Python packages
    git \              # Required if any repo dependencies need to be cloned
    && apt-get clean \  # Clean up package cache
    && rm -rf /var/lib/apt/lists/*  # Free up space by removing unnecessary files

# Copy the Python requirements file to the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the FastAPI port (8000) so it can be accessed outside the container
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
