from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import Classifier
from pyspark.sql.functions import col, lit, when, mean
import numpy as np
import json


def gini_impurity(positive, total):
    """Calculate Gini impurity."""
    if total == 0:
        return 0
    negative = total - positive
    prob_positive = positive / total
    prob_negative = negative / total
    return 1 - prob_positive**2 - prob_negative**2


def entropy(positive, total):
    """Calculate entropy."""
    if total == 0 or positive == 0 or positive == total:
        return 0
    prob_positive = positive / total
    prob_negative = 1 - prob_positive
    return -(
        prob_positive * np.log2(prob_positive) + prob_negative * np.log2(prob_negative)
    )


class PySparkMADTClassifier(Classifier):
    """Missingness-avoiding decision tree classifier."""

    def __init__(
        self,
        criterion="gini",
        maxDepth=3,
        alpha=1.0,
        seed=None,
        labelCol="credit_approved",
    ):
        super(PySparkMADTClassifier, self).__init__()
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.alpha = alpha
        self.seed = seed
        self.labelCol = labelCol

    def _fit(self, dataset):
        """Fit the PySparkMADTClassifier."""
        # Add default sample weights if not already present
        if "sample_weight" not in dataset.columns:
            dataset = dataset.withColumn("sample_weight", lit(1.0))

        # Initialize tree
        self.tree_ = self._build_tree(dataset, depth=0)

    def _build_tree(self, dataset, depth):
        """Recursively build the decision tree."""
        label_col = self.labelCol

        # Stop conditions
        if depth >= self.maxDepth or dataset.count() == 0:
            return self._get_leaf(dataset)

        # Check if node is homogeneous
        label_counts = dataset.groupBy(label_col).count().collect()
        if len(label_counts) == 1:
            return {"Predict": label_counts[0][label_col]}

        # Find the best split
        best_feature, best_threshold, best_score = None, None, float("-inf")
        for feature in dataset.columns:
            if (
                feature != label_col
                and feature != "sample_weight"
                and not feature.startswith("missing_")
            ):
                score, threshold = self._best_split(dataset, feature, label_col)
                if score > best_score:
                    best_feature, best_threshold, best_score = feature, threshold, score

        if best_feature is None:
            return self._get_leaf(dataset)

        # Split the dataset
        left_data = dataset.filter(col(best_feature) <= lit(best_threshold))
        right_data = dataset.filter(col(best_feature) > lit(best_threshold))

        return {
            "If": f"{best_feature} <= {best_threshold}",
            "Left": self._build_tree(left_data, depth + 1),
            "Else": f"{best_feature} > {best_threshold}",
            "Right": self._build_tree(right_data, depth + 1),
        }

    def _get_leaf(self, dataset):
        """Get the prediction for a leaf node."""
        if dataset.count() == 0:
            return {"Predict": None}

        label_col = self.labelCol
        prediction = (
            dataset.groupBy(label_col)
            .count()
            .orderBy("count", ascending=False)
            .first()[label_col]
        )
        return {"Predict": prediction}

    def _best_split(self, dataset, feature, label_col):
        """Find the best split for a feature, including missingness penalty."""
        # Collect data sorted by feature
        sorted_data = (
            dataset.select(feature, label_col, "sample_weight")
            .filter(col(feature).isNotNull())
            .orderBy(feature)
            .collect()
        )

        if not sorted_data:
            return float("-inf"), None

        # Compute penalties for missing data using the missingness mask
        missing_feature_col = f"missing_{feature}"
        if missing_feature_col in dataset.columns:
            missing_penalty = (
                dataset.select(missing_feature_col, "sample_weight")
                .rdd.map(lambda row: row[missing_feature_col] * row["sample_weight"])
                .sum()
                * self.alpha
            )
        else:
            missing_penalty = 0

        # Aggregate weights and labels for computing split criteria
        total_weight = sum(row["sample_weight"] for row in sorted_data)
        total_positive = sum(
            row["sample_weight"] for row in sorted_data if row[label_col] == 1
        )

        best_score = float("-inf")
        best_threshold = None

        weight_left = 0
        positive_left = 0

        for i in range(len(sorted_data) - 1):
            weight_left += sorted_data[i]["sample_weight"]
            positive_left += (
                sorted_data[i]["sample_weight"] if sorted_data[i][label_col] == 1 else 0
            )

            weight_right = total_weight - weight_left
            positive_right = total_positive - positive_left

            if weight_left == 0 or weight_right == 0:
                continue

            if self.criterion == "gini":
                gini_left = gini_impurity(positive_left, weight_left)
                gini_right = gini_impurity(positive_right, weight_right)
                score = -(
                    (weight_left / total_weight) * gini_left
                    + (weight_right / total_weight) * gini_right
                )
            elif self.criterion == "entropy":
                entropy_left = entropy(positive_left, weight_left)
                entropy_right = entropy(positive_right, weight_right)
                score = -(
                    (weight_left / total_weight) * entropy_left
                    + (weight_right / total_weight) * entropy_right
                )
            else:
                raise ValueError("Unsupported criterion")

            # Add missingness penalty
            score -= missing_penalty

            if score > best_score:
                best_score = score
                best_threshold = (
                    sorted_data[i][feature] + sorted_data[i + 1][feature]
                ) / 2

        return best_score, best_threshold

    def predict(self, dataset):
        """Predict using the trained tree."""

        def traverse_tree(row, node):
            if "Predict" in node:
                return node["Predict"]
            condition = node["If"]
            feature = condition.split()[0]
            operator = condition.split()[1]
            threshold = float(condition.split()[2])

            if operator == "<=":
                if row[feature] <= threshold:
                    return traverse_tree(row, node["Left"])
                else:
                    return traverse_tree(row, node["Right"])
            elif operator == ">":
                if row[feature] > threshold:
                    return traverse_tree(row, node["Right"])
                else:
                    return traverse_tree(row, node["Left"])

        return dataset.rdd.map(
            lambda row: traverse_tree(row.asDict(), self.tree_)
        ).collect()

    def save(self, path):
        """Save the trained model."""
        import json

        with open(path, "w") as f:
            json.dump(self.tree_, f)

    def print_tree(self):
        """Pretty print the decision tree."""

        def recurse(node, depth):
            indent = "  " * depth
            if "Predict" in node:
                print(f"{indent}Predict: {node['Predict']}")
            else:
                print(f"{indent}If ({node['If']})")
                recurse(node["Left"], depth + 1)
                print(f"{indent}Else ({node['Else']})")
                recurse(node["Right"], depth + 1)

        recurse(self.tree_, 0)

    def compute_missingness_reliance(self, dataset):
        """
        Compute the fraction of rows where missing features influence the decision path.

        Args:
            dataset (DataFrame): Input PySpark DataFrame with missingness masks.

        Returns:
            float: Proportion of rows influenced by missing features.
        """
        tree = self.tree_

        def traverse_and_check_missing(row, node):
            """Traverse the tree and check for missing feature reliance."""
            if "Predict" in node:
                return 0  # Leaf node, no missing feature reliance

            # Extract decision condition
            condition = node["If"]
            feature = condition.split()[0]
            threshold = float(condition.split()[2])

            # Check if the feature is missing
            is_missing = row.get(f"missing_{feature}", 0)  # Missingness mask column

            # Traverse the next node
            if row[feature] is not None and row[feature] <= threshold:
                return is_missing or traverse_and_check_missing(row, node["Left"])
            else:
                return is_missing or traverse_and_check_missing(row, node["Right"])

        # Count missing reliance over all rows
        missing_reliance_count = dataset.rdd.map(
            lambda row: traverse_and_check_missing(row.asDict(), tree)
        ).sum()

        total_rows = dataset.count()
        return missing_reliance_count / total_rows if total_rows > 0 else 0


class PySparkMARFClassifier:
    """Missingness-avoiding random forest classifier."""

    def __init__(
        self,
        numTrees=100,
        criterion="gini",
        maxDepth=10,
        bootstrap=True,
        seed=None,
        alpha=1.0,
        labelCol="credit_approved",
    ):
        self.numTrees = numTrees
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.bootstrap = bootstrap
        self.seed = seed
        self.alpha = alpha
        self.labelCol = labelCol
        self.trees = []

    def _fit(self, dataset):
        """Fit the PySparkMARFClassifier."""
        self.trees = []

        for i in range(self.numTrees):
            if self.bootstrap:
                sample = dataset.sample(
                    withReplacement=True, fraction=1.0, seed=self.seed
                )
            else:
                sample = dataset

            tree = PySparkMADTClassifier(
                criterion=self.criterion,
                maxDepth=self.maxDepth,
                alpha=self.alpha,
                seed=self.seed,
                labelCol=self.labelCol,
            )
            tree._fit(sample)
            self.trees.append(tree)

    def predict(self, dataset):
        """Predict using the trained forest."""
        # Extract the tree structures and broadcast them
        tree_structures = [tree.tree_ for tree in self.trees]
        broadcast_trees = dataset._sc.broadcast(tree_structures)

        def traverse_tree(row, tree):
            """Recursively traverse a tree to make a prediction."""
            if "Predict" in tree:
                return tree["Predict"]
            condition = tree["If"]
            feature = condition.split()[0]
            operator = condition.split()[1]
            threshold = float(condition.split()[2])

            if operator == "<=":
                if row[feature] <= threshold:
                    return traverse_tree(row, tree["Left"])
                else:
                    return traverse_tree(row, tree["Right"])
            elif operator == ">":
                if row[feature] > threshold:
                    return traverse_tree(row, tree["Right"])
                else:
                    return traverse_tree(row, tree["Left"])

        def aggregate_predictions(row):
            """Aggregate predictions from all trees (majority voting)."""
            predictions = [
                traverse_tree(row.asDict(), tree) for tree in broadcast_trees.value
            ]
            return max(set(predictions), key=predictions.count)

        # Apply predictions for all rows
        return dataset.rdd.map(aggregate_predictions).collect()

    def save(self, path):
        """Save the forest model."""
        import json

        with open(path, "w") as f:
            trees_serialized = [tree.tree_ for tree in self.trees]
            json.dump(trees_serialized, f)

    def print_forest(self):
        """Print all decision trees in the forest."""
        for i, tree in enumerate(self.trees):
            print(f"Tree {i + 1}:")
            tree.print_tree()
            print("\n")

    def compute_missingness_reliance(self, dataset):
        """
        Compute the fraction of rows where missing features influence the decision path
        across all trees in the random forest.

        Args:
            dataset (DataFrame): Input PySpark DataFrame with missingness masks.

        Returns:
            float: Average missingness reliance over all trees.
        """
        # Compute missingness reliance for each tree
        reliance_list = [
            tree.compute_missingness_reliance(dataset) for tree in self.trees
        ]

        # Average the reliance values
        if len(reliance_list) == 0:
            return 0
        return sum(reliance_list) / len(reliance_list)


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
    data_path = "./src/data/sample_data.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Define features and label
    features = ["credit_age", "income", "full_time", "part_time", "student"]
    label = "credit_approved"
    alpha = 0.1
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
        criterion="gini", maxDepth=3, alpha=alpha, labelCol=label
    )
    madt._fit(df)
    print("Trained MADT Tree:")
    madt.print_tree()

    # Make Predictions
    predictions = madt.predict(df)
    print("MADT Predictions:", predictions)

    # Compute missingness reliance for MADT
    reliance = madt.compute_missingness_reliance(df)
    print(f"Missingness reliance for MADT: {reliance:.4f}")

    madt.save("./src/models/madt_model")

    # Train PySparkMARFClassifier
    marf = PySparkMARFClassifier(
        numTrees=3, criterion="gini", maxDepth=3, alpha=alpha, labelCol=label
    )
    marf._fit(df)
    print("Trained MARF Forest:")
    marf.print_forest()

    # Make Predictions with MARF
    forest_predictions = marf.predict(df)
    print("MARF Predictions:", forest_predictions)

    # Compute missingness reliance for MARF
    forest_reliance = marf.compute_missingness_reliance(df)
    print(f"Missingness reliance for MARF: {forest_reliance:.4f}")

    marf.save("./src/models/marf_model")

    spark.stop()
