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

from matrees.estimators import MADTClassifier, PySparkMADTClassifier

FRAC_INFORMATIVE = 0.2

FRAC_REDUNDANT = 0.2


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

    missingness = rng.uniform(missingness_low, missingness_high, n_features)

    for i, m in enumerate(missingness):
        M[:, i] = rng.uniform(0, 1, n_samples) < m

    X = X.mask(M)

    missing_columns = [f"missing_{f}" for f in feature_columns]
    M = pd.DataFrame(M, columns=missing_columns)

    return X, M, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a MA tree classifier to synthetic data.")
    parser.add_argument(
        "--estimator_alias", type=str, required=True,
        choices=["madt", "spark_madt", "spark_madt_scala"], help="The estimator to fit.",
    )
    parser.add_argument("--n_train", type=int, default=100, help="Number of train samples.")
    parser.add_argument("--n_test", type=int, default=100, help="Number of test samples.")
    parser.add_argument("--n_features", type=int, default=20, help="Number of features.")
    parser.add_argument("--max_depth", type=int, default=5, help="The maximum depth of the tree.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Missingness regularization parameter.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=None)
    args = parser.parse_args()

    n_informative = int(FRAC_INFORMATIVE * args.n_features)
    n_redundant = int(FRAC_REDUNDANT * args.n_features)

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

    # Perform zero imputation.
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    if args.estimator_alias == "madt":
        estimator = MADTClassifier(
            max_depth=args.max_depth,
            alpha=args.alpha,
            random_state=args.seed,
        )

        start_time = time.time()
        estimator.fit(X_train, y_train, M=M_train)
        end_time = time.time()

        yp_test = estimator.predict(X_test)

        rho_test = estimator.compute_missingness_reliance(X_test, M_test)

    elif args.estimator_alias == "spark_madt":
        spark = (
            SparkSession.builder.appName("DecisionTreeClassifier")
            .master("local[*]")
            .config("spark.driver.memory", "8g")
            .getOrCreate()
        )

        df_train = pd.concat([X_train, M_train, y_train], axis=1)
        df_test = pd.concat([X_test, M_test, y_test], axis=1)

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

        yp_test = estimator.predict(df_test)

        rho_test = estimator.compute_missingness_reliance(df_test)

        spark.stop()

    elif args.estimator_alias == "spark_madt_scala":
        spark = (
            SparkSession.builder.appName("DecisionTreeClassifier")
            .master("local[*]")
            .config("spark.driver.memory", "8g")
            .getOrCreate()
        )

        df_train = pd.concat([X_train, M_train, y_train], axis=1)
        df_test = pd.concat([X_test, M_test, y_test], axis=1)

        df_train = spark.createDataFrame(df_train)
        df_test = spark.createDataFrame(df_test)

        assembler1 = VectorAssembler(inputCols=X.columns.tolist(), outputCol="features")
        assembler2 = VectorAssembler(inputCols=M.columns.tolist(), outputCol="missing")

        df_train = assembler1.transform(df_train)
        df_train = assembler2.transform(df_train)

        df_test = assembler1.transform(df_test)
        df_test = assembler2.transform(df_test)

        estimator = DecisionTreeClassifier(
            impurity="gini",
            maxDepth=args.max_depth,
            seed=args.seed,
            featuresCol="features",
            labelCol="label",
            missingnessCol="missing",
            alpha=args.alpha,
            missingnessRelianceCol="rho",
        )

        start_time = time.time()
        estimator = estimator.fit(df_train)
        end_time = time.time()

        df_test = estimator.transform(df_test)

        yp_test = df_test.select("prediction") \
            .rdd.flatMap(lambda x: x).collect()

        rho_test = np.mean(
            df_test.select("rho").rdd.flatMap(lambda x: x).collect()
        )

        spark.stop()

    training_time = end_time - start_time

    accuracy_test = accuracy_score(y_test, yp_test)

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
            "accuracy": accuracy_test,
            "rho": rho_test,
        }
    )

    if args.task_id is None:
        output_file = f"results_{args.estimator_alias}.csv"
    else:
        output_file = f"{args.task_id:02d}_results_{args.estimator_alias}.csv"

    results.to_csv(join(args.output_dir, output_file), header=False)
