# Project title: Scalable Missingess-Avoiding Decision Trees
 - Newton Mwai Kinyanjui, Data Science and AI Division, Computer Science and Engineering Department, Chalmers University of Technology
 - Anton Mattson, Data Science and AI Division, Computer Science and Engineering Department, Chalmers University of Technology
## Brief description of the project

This project focuses on developing decision trees that avoid relying on missing values during prediction tasks, which is particularly important for safety-critical and interpretable machine learning (ML) applications like healthcare. Missing values often complicate model deployment in prediction tasks, and traditional imputation methods can undermine interpretability.

The core idea is to regularize the node-splitting criterion in decision trees to minimize the presence of missing values on decision paths. The project explores scalable implementations of Missingness-Avoiding (MA) Trees using Apache Spark to efficiently handle large datasets. The work compares three implementations:
- Python with scikit-learn
- PySpark with Apache Spark MLlib
- Scala with Apache Spark MLlib (by modifying Spark source code)

## Link to presentation
 - https://docs.google.com/presentation/d/1jKvugr8AxkjlyV_a55xIcfPLEN_FhEYUKjQ92o0WiYw/edit?usp=sharing

A brief description of Authors' Contributions
 - Newton
   - Idea brainstorming of the solution approach for MA-Trees with Apache Spark
   - Set up the development pipeline with PySpark and Apache MLLib
   - Wrote the Pyspark implementations of MADTClassifier and MARFClassifier
   - Wrote shell scripts for running the experiments locally with Apache PySpark and Docker
   - Brainstormed changing the Spark Scala source code, locally debugged and built Scala source code
   - Code organization to fit the experiment pipeline
   - Prepared the slides for the project pitch and project presentation
    
- Anton
   - 
