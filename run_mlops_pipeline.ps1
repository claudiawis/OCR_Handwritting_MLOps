Write-Output "Starting MLOps pipeline execution..."

# Run ingestion service
Write-Output "Running ingestion service..."
docker-compose up -d ingestion
Start-Sleep -Seconds 10  # Wait for ingestion to start

# Run training service
Write-Output "Running training service..."
docker-compose up -d training
Start-Sleep -Seconds 10  # Wait for training to start

# Run prediction service
Write-Output "Running prediction service..."
docker-compose up -d prediction
Start-Sleep -Seconds 10

# Start monitoring services
Write-Output "Starting monitoring services..."
docker-compose up -d prometheus grafana

Write-Output "MLOps pipeline execution completed."
