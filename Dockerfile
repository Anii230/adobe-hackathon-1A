# Use slim Python 3.10 base
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyMuPDF, spaCy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy spaCy models tarballs and install
COPY models/en_core_web_sm-3.8.0/dist/en_core_web_sm-3.8.0.tar.gz ./models/
COPY models/xx_sent_ud_sm-3.8.0/dist/xx_sent_ud_sm-3.8.0.tar.gz ./models/

RUN pip install --no-cache-dir ./models/en_core_web_sm-3.8.0.tar.gz && \
    pip install --no-cache-dir ./models/xx_sent_ud_sm-3.8.0.tar.gz && \
    rm -f ./models/*.tar.gz

# Copy the rest of the codebase
COPY . .

# Run main script
CMD ["python", "main.py"]
