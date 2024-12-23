# Scalable missingess-avoiding decision trees
This project was completed as part of the WASP course [Scalable Data Science and Distributed Machine Learning](https://lamastex.github.io/scalable-data-science/sds/3/x/) during the fall 2024 semester. The project was collaboratively developed by two contributors: 
 1. **Newton Mwai Kinyanjui**, Data Science and AI Division, CSE Department, Chalmers University of Technology
 2. **Anton Matsson**, Data Science and AI Division, CSE Department, Chalmers University of Technology
   
## Brief project description

This project focuses on developing decision trees designed to avoid reliance on missing values during prediction tasks. Such missingness-avoiding (MA) trees are particularly valuable for safety-critical and interpretable machine learning applications like healthcare. Missing values often complicate model deployment, and traditional imputation methods can undermine interpretability.

The core idea of this project is to regularize the node-splitting criterion in decision trees to minimize the presence of missing values along decision paths. To ensure scalability, the project explores implementations of missingness-avoiding (MA) trees using Apache Spark, enabling efficient handling of large datasets. Specifically, it compares three different implementations of an MA tree for classification:
- [`MADTClassifier`](https://github.com/antmats/matrees/blob/main/matrees/estimators.py#L277): A pure Python-based implementation.
- [`PySparkMADTClassifier`](https://github.com/antmats/matrees/blob/main/matrees/estimators.py#L373): A Python-based implementation utilizing RDD operations in PySpark.
- A modified implementation of the Scala [`DecisionTreeClassifier`](https://github.com/antmats/spark/blob/matrees/mllib/src/main/scala/org/apache/spark/ml/classification/DecisionTreeClassifier.scala) class in Spark MLlib, accompanied by a corresponding [Python API](https://github.com/antmats/spark/blob/matrees/python/pyspark/ml/classification.py#L1692).

We compare the different implementations using a synthetic dataset with randomly introduced missing values. Training times are measured for each implementation while varying the number of samples from 100 to 1,000,000, using 20 features. Additionally, we increase the number of features from 10 to 100,000 while keeping the number of samples fixed at 1,000. Finally, we vary the missingness regularization parameter and compare accuracy and missingness reliance across implementations. The results of the experiments are presented in the notebook [`results.ipynb`](results.ipynb).

## Project presentation

A link to the presentation of the project can be found [here](https://docs.google.com/presentation/d/1jKvugr8AxkjlyV_a55xIcfPLEN_FhEYUKjQ92o0WiYw/edit?usp=sharing).

## Authors' contributions
**Newton:**
- Brainstormed the approach for implementing MA trees using Apache Spark
- Set up the development pipeline with PySpark and Spark MLlib
- Implemented the `PySparkMADTClassifier`
- Wrote shell scripts for running the experiments locally with PySpark and Docker
- Brainstormed changes to the Spark Scala source code, then locally built and debugged the modified Scala source code
- Organized code to align with the experiment pipeline
- Prepared slides for the project pitch and the project presentation
    
**Anton:**
-  Brainstormed the approach for implementing MA trees using Apache Spark
-  Implemented te `MADTClassifier`
-  Implemented the MA trees in Scala by changing the Spark source code
-  Set up comparison experiment for the implementations
-  Set up experiment pipeline with synthetic data on HPC cluster
-  Organized code to align with the experiment pipeline
-  Prepared slides for the project pitch and the project presentation
