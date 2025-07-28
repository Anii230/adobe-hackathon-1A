# Stage 1: Use an official Python runtime as a parent image
# Specify platform for amd64 architecture and use a slim version for a smaller size.
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
# This is done first to leverage Docker's layer caching.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures the image is smaller
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy models required by your application for offline use.
# This is a critical step for meeting the "no network" constraint.
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download ja_core_news_sm

# Copy the rest of your application's source code from your host to your image filesystem.
COPY . .

# Specify the command to run on container startup.
# This will execute your main script.
CMD ["python", "main.py"]