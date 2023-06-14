# Use the official Python base image with version 3.8
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Install dependencies required for dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application directory into the container
COPY . .

# Set the default command to run when the container starts
CMD ["python", "app.py"]
