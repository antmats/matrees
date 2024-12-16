# Use the bitnami/spark image as the base image
FROM bitnami/spark:latest

# Install Python libraries required for PySpark
RUN pip install numpy pandas networkx scikit-learn 
# matplotlib seaborn

# Set the working directory
WORKDIR /opt/bitnami/spark
