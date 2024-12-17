import argparse
import time
from os.path import join

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.functions import udf, col
from pyspark.ml.linalg import DenseVector, VectorUDT

from matrees.estimators import MADTClassifier, PySparkMADTClassifier


def get_datatset(
    n_samples,
    n_features=20,
    n_informative=2,
    n_redundant=2,
    n_classes=2,
    missingness_low=0.05,
    missingness_high=0.4,
    seed=None,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=seed,
    )

    feature_columns = [f"x_{i}" for i in range(n_features)]
    X = pd.DataFrame(X, columns=feature_columns)
    y = pd.Series(y, name="label")
    
    rng = np.random.default_rng(seed)

    M = np.zeros_like(X, dtype=bool)

    missingness = rng.uniform(
        missingness_low, missingness_high, n_features
    )

    for i, m in enumerate(missingness):
        M[:, i] = rng.uniform(0, 1, n_samples) < m

    X = X.mask(M)

    missing_columns = [f"missing_{f}" for f in feature_columns]
    M = pd.DataFrame(M, columns=missing_columns)

    return X, M, y


def to_dense(vector):
    return DenseVector(vector.toArray())


to_dense_udf = udf(to_dense, VectorUDT())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator_alias", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--n_features", type=int, default=20)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=None)
    args = parser.parse_args()

    n_informative = n_redundant = int(0.2 * args.n_features)

    n_samples = args.n_train + args.n_test
    X, M, y = get_datatset(
        n_samples,
        n_features=args.n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        seed=args.seed,
    )

    X_train, X_test, M_train, M_test, y_train, y_test = train_test_split(
        X, M, y, test_size=args.n_test, random_state=args.seed
    )

    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    if args.estimator_alias == "madt":
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        
        M_train = M_train.to_numpy()
        M_test = M_test.to_numpy()

        estimator = MADTClassifier(
            max_depth=args.max_depth,
            alpha=args.alpha,
            random_state=args.seed,
        )

        start_time = time.time()
        estimator.fit(X_train, y_train, M=M_train)
        end_time = time.time()

        test_predictions = estimator.predict(X_test)

    elif args.estimator_alias == "spark_madt":
        df_train = pd.concat([X_train, M_train, y_train], axis=1)
        df_test = pd.concat([X_test, M_test, y_test], axis=1)

        spark = SparkSession.builder \
            .appName("DecisionTreeModel") \
            .master("local[*]") \
            .getOrCreate()

        df_train = spark.createDataFrame(df_train)
        df_test = spark.createDataFrame(df_test)

        estimator = PySparkMADTClassifier(
            criterion="gini",
            maxDepth=args.max_depth,
            alpha=args.alpha,
            labelCol="label",
        )

        start_time = time.time()
        estimator._fit(df_train)
        end_time = time.time()

        test_predictions = estimator.predict(df_test)

        spark.stop()

    elif args.estimator_alias == "spark_madt_scala":
        df_train = pd.concat([X_train, M_train, y_train], axis=1)
        df_test = pd.concat([X_test, M_test, y_test], axis=1)

        spark = SparkSession.builder \
            .appName("DecisionTreeModel") \
            .master("local[*]") \
            .getOrCreate()

        df_train = spark.createDataFrame(df_train)
        df_test = spark.createDataFrame(df_test)

        assembler1 = VectorAssembler(
            inputCols=X.columns.tolist(), outputCol="features"
        )
        assembler2 = VectorAssembler(
            inputCols=M.columns.tolist(), outputCol="missing"
        )

        df_train = assembler1.transform(df_train)
        df_train = assembler2.transform(df_train)

        df_test = assembler1.transform(df_test)

        estimator = DecisionTreeClassifier(
            impurity="gini",
            maxDepth=args.max_depth,
            seed=args.seed,
            featuresCol="features",
            labelCol="label",
            missingnessCol="missing",
            alpha=args.alpha,
        )

        start_time = time.time()
        estimator = estimator.fit(df_train)
        end_time = time.time()

        test_predictions = estimator.transform(df_test) \
            .select("prediction").rdd.flatMap(lambda x: x).collect()

        spark.stop()

    training_time = end_time - start_time

    accuracy = accuracy_score(y_test, test_predictions)

    results = pd.Series(
        {
            "estimator_alias": args.estimator_alias,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_features": args.n_features,
            "max_depth": args.max_depth,
            "alpha": args.alpha,
            "seed": args.seed,
            "training_time": training_time,
            "accuracy": accuracy,
        }
    )
    
    if args.task_id is None:
        output_file = f"results_{args.estimator_alias}.csv"
    else:
        output_file = f"{args.task_id:02d}_results_{args.estimator_alias}.csv"
    
    results.to_csv(join(args.output_dir, output_file), header=False)
