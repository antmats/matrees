#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Build Docker image
echo "Building Docker image..."
docker compose build

# Start Spark cluster
echo "Starting Spark cluster..."
docker compose up -d

# Wait for Spark master to start
echo "Waiting for Spark master to start..."
sleep 10

# Get Spark master IP using docker inspect
SPARK_MASTER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' pyspark_matrees-spark-master-1)
echo "Spark master IP: $SPARK_MASTER_IP"
if [ -z "$SPARK_MASTER_IP" ]; then
    echo "Error: Unable to retrieve Spark master IP."
    docker compose down
    exit 1
fi
echo "Spark master is running at spark://$SPARK_MASTER_IP:7077"

# Copy Python script to Spark master
echo "Copying training script to Spark master..."
docker cp -L ./src/main.py pyspark_matrees-spark-master-1:/opt/bitnami/spark/main.py
docker cp -L ./src/estimators.py pyspark_matrees-spark-master-1:/opt/bitnami/spark/estimators.py
docker cp -L ./src/utils.py pyspark_matrees-spark-master-1:/opt/bitnami/spark/utils.py

# Execute training script using spark-submit
echo "Running training script with Spark master..."
docker compose exec spark-master spark-submit \
    --master spark://$SPARK_MASTER_IP:7077 \
    /opt/bitnami/spark/main.py

# Stop the Spark cluster
echo "Stopping Spark cluster..."
docker compose down

echo "Done!"
