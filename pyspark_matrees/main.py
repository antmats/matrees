from pyspark.ml.classification import Classifier
from pyspark.sql.functions import col, lit, when, mean

from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, when
from pyspark.ml.feature import VectorAssembler
import numpy as np
import json

from estimators import PySparkMADTClassifier, PySparkMARFClassifier
from utils import *

if __name__ == "__main__":

    # Create Spark Session
    spark = SparkSession.builder.appName("MissingnessAvoidingPyspark").getOrCreate()

    # # Read the original CSV file
    # csv_path = "./data/nhanes.csv"
    # df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # # Partition and save the CSV file
    # output_path = "./data/partitioned_nhanes.csv"
    # partition_and_save_csv(df, output_path, num_partitions=8)

    # print(f"CSV file successfully partitioned and saved at: {output_path}")

    # Load Dataset
    data_path = "./data/sample_data.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Define features and label
    features = ["credit_age", "income", "full_time", "part_time", "student"]
    label = "credit_approved"

    # Add Missingness Mask and Impute Missing Values
    for feature in features:
        df = df.withColumn(
            f"missing_{feature}", when(col(feature).isNull(), 1.0).otherwise(0.0)
        )
        mean_value = df.select(mean(col(feature))).first()[0]
        df = df.withColumn(
            feature,
            when(col(feature).isNull(), lit(mean_value)).otherwise(col(feature)),
        )

    # Train PySparkMADTClassifier
    madt = PySparkMADTClassifier(
        criterion="gini", maxDepth=3, alpha=1.0, labelCol=label
    )
    madt._fit(df)
    print("Trained MADT Tree:")
    madt.print_tree()

    # Make Predictions
    predictions = madt.predict(df)
    print("MADT Predictions:", predictions)
    madt.save("./models/madt_model")

    # Train PySparkMARFClassifier
    marf = PySparkMARFClassifier(
        numTrees=3, criterion="gini", maxDepth=3, alpha=1.0, labelCol=label
    )
    marf._fit(df)
    print("Trained MARF Forest:")
    marf.print_forest()

    # Make Predictions with MARF
    forest_predictions = marf.predict(df)
    print("MARF Predictions:", forest_predictions)
    marf.save("./models/marf_model")

    spark.stop()
