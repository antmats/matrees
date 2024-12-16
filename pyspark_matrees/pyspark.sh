#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Build Docker image
echo "Building Docker image..."
docker compose build

# Start Spark cluster
echo "Starting Spark cluster..."
docker compose up -d

# Wait for Spark containers to start
echo "Waiting for Spark master to start..."
sleep 5

# Function to get Spark master container name
get_spark_master_name() {
    docker ps --filter "name=spark-master" --format "{{.Names}}" | head -n 1
}

# Retry logic to wait for Spark master
RETRIES=10
SPARK_MASTER_NAME=""
for i in $(seq 1 $RETRIES); do
    echo "Checking for Spark master container... (Attempt $i/$RETRIES)"
    SPARK_MASTER_NAME=$(get_spark_master_name)
    if [ -n "$SPARK_MASTER_NAME" ]; then
        echo "Found Spark master container: $SPARK_MASTER_NAME"
        break
    fi
    sleep 5
done

if [ -z "$SPARK_MASTER_NAME" ]; then
    echo "Error: Spark master container did not start. Exiting..."
    docker compose down
    exit 1
fi

# Retrieve Spark master IP
SPARK_MASTER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$SPARK_MASTER_NAME")

if [ -z "$SPARK_MASTER_IP" ]; then
    echo "Error: Unable to retrieve Spark master IP."
    docker compose down
    exit 1
fi
echo "Spark master is running at spark://$SPARK_MASTER_IP:7077"

# Copy scripts to Spark master dynamically
echo "Copying scripts to Spark master ($SPARK_MASTER_NAME)..."
docker cp -L ./main.py "$SPARK_MASTER_NAME":/opt/bitnami/spark/main.py
docker cp -L ./estimators.py "$SPARK_MASTER_NAME":/opt/bitnami/spark/estimators.py
docker cp -L ./utils.py "$SPARK_MASTER_NAME":/opt/bitnami/spark/utils.py

# Run Spark job using spark-submit
echo "Running training script on Spark master..."
docker exec -it "$SPARK_MASTER_NAME" spark-submit \
    --master spark://$SPARK_MASTER_IP:7077 \
    /opt/bitnami/spark/main.py

# Stop the Spark cluster
echo "Stopping Spark cluster..."
docker compose down

echo "Done!"
