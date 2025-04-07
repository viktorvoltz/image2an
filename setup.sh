#!/bin/bash

# Create directory structure
mkdir -p static

# Copy frontend HTML to static folder
cp index.html static/

# Build Docker image
docker build -t anime-converter .

# Run Docker container
docker run -p 5000:5000 anime-converter