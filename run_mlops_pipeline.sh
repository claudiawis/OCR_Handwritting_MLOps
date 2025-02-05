#!/bin/bash

echo "Starting MLOps pipeline execution..." 

# Run ingestion service
echo "Running ingestion service..."
docker-compose up -d ingestion

# Wait for ingestion to finish
docker wait ingestion_service

# Run training service
echo "Running training service..."
docker-compose up -d training

# Wait for training to finish
docker wait training_service

# Run prediction service
echo "Running prediction service..."
docker-compose up -d prediction

# Wait for prediction service to finish (if needed)
docker wait prediction_service

# Start monitoring services (Prometheus and Grafana)
echo "Starting monitoring services..."
docker-compose up -d prometheus grafana

echo "MLOps pipeline execution completed."
