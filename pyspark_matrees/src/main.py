from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, sum as spark_sum, mean, asc
import json


def gini_impurity(positive, total):
    """Calculate Gini impurity."""
    if total == 0:
        return 0
    negative = total - positive
    prob_positive = positive / total
    prob_negative = negative / total
    return 1 - prob_positive**2 - prob_negative**2


class PySparkMADTClassifier:
    """Missingness-avoiding decision tree classifier."""

    def __init__(self, criterion="gini", maxDepth=3, alpha=1.0, labelCol="label"):
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.alpha = alpha
        self.labelCol = labelCol
        self.tree_ = None

    def _fit(self, dataset):
        """Fit the decision tree on the dataset."""
        if "sample_weight" not in dataset.columns:
            dataset = dataset.withColumn("sample_weight", lit(1.0))
        self.tree_ = self._build_tree(dataset, depth=0)

    def _build_tree(self, dataset, depth):
        """Recursively build the decision tree."""
        if depth >= self.maxDepth or dataset.count() == 0:
            return self._get_leaf(dataset)

        # Check for homogeneity
        label_counts = (
            dataset.groupBy(self.labelCol)
            .agg(spark_sum("sample_weight").alias("weight"))
            .collect()
        )
        if len(label_counts) == 1:
            return {"Predict": label_counts[0][self.labelCol]}

        # Find the best split using histogram bins
        best_feature, best_threshold, best_score = None, None, float("-inf")
        for feature in dataset.columns:
            if feature not in [
                self.labelCol,
                "sample_weight",
            ] and not feature.startswith("missing_"):
                score, threshold = self._best_split(dataset, feature)
                if score is not None and score > best_score:
                    best_feature, best_threshold, best_score = feature, threshold, score

        if best_feature is None:
            return self._get_leaf(dataset)

        # Split the dataset
        left_data = dataset.filter(col(best_feature) <= best_threshold)
        right_data = dataset.filter(col(best_feature) > best_threshold)

        return {
            "If": f"{best_feature} <= {best_threshold}",
            "Left": self._build_tree(left_data, depth + 1),
            "Else": f"{best_feature} > {best_threshold}",
            "Right": self._build_tree(right_data, depth + 1),
        }

    def _best_split(self, dataset, feature):
        """Find the best split for a feature using histogram bins."""
        # Missing data penalty
        missing_penalty = (
            dataset.filter(col(f"missing_{feature}") == 1)
            .agg(spark_sum("sample_weight"))
            .collect()[0][0]
            or 0
        ) * self.alpha

        # Create histogram bins
        histogram = (
            dataset.groupBy(feature)
            .agg(
                spark_sum("sample_weight").alias("total_weight"),
                spark_sum(
                    when(col(self.labelCol) == 1, col("sample_weight")).otherwise(0)
                ).alias("positive_weight"),
            )
            .orderBy(asc(feature))
        ).collect()

        if len(histogram) < 2:
            return None, None

        # Initialize cumulative sums
        total_weight = sum(row["total_weight"] for row in histogram)
        total_positive = sum(row["positive_weight"] for row in histogram)

        weight_left, positive_left = 0, 0
        best_score, best_threshold = float("-inf"), None

        for i in range(len(histogram) - 1):  # Iterate through bin boundaries
            weight_left += histogram[i]["total_weight"]
            positive_left += histogram[i]["positive_weight"]

            weight_right = total_weight - weight_left
            positive_right = total_positive - positive_left

            if weight_left == 0 or weight_right == 0:
                continue

            # Calculate Gini impurity for left and right splits
            gini_left = gini_impurity(positive_left, weight_left)
            gini_right = gini_impurity(positive_right, weight_right)

            # Weighted Gini score
            gini_score = (
                -(
                    (weight_left / total_weight) * gini_left
                    + (weight_right / total_weight) * gini_right
                )
                - missing_penalty
            )

            # Update the best score and threshold
            if gini_score > best_score:
                best_score = gini_score
                best_threshold = (histogram[i][feature] + histogram[i + 1][feature]) / 2

        return best_score, best_threshold

    def _get_leaf(self, dataset):
        """Return the majority class at a leaf node."""
        result = (
            dataset.groupBy(self.labelCol)
            .agg(spark_sum("sample_weight").alias("weight"))
            .orderBy("weight", ascending=False)
            .first()
        )
        return {"Predict": result[self.labelCol]} if result else {"Predict": None}

    def predict(self, dataset):
        """Predict using the trained decision tree."""

        def traverse_tree(row, node):
            if "Predict" in node:
                return node["Predict"]
            feature, operator, threshold = node["If"].split()
            threshold = float(threshold)
            if row[feature] <= threshold:
                return traverse_tree(row, node["Left"])
            else:
                return traverse_tree(row, node["Right"])

        return dataset.rdd.map(
            lambda row: traverse_tree(row.asDict(), self.tree_)
        ).collect()

    def compute_missingness_reliance(self, dataset):
        """Compute the proportion of rows influenced by missing features."""

        def traverse_and_check_missing(row, node):
            if "Predict" in node:
                return 0
            feature = node["If"].split()[0]
            is_missing = row.get(f"missing_{feature}", 0)
            if row[feature] is not None:
                if row[feature] <= float(node["If"].split()[2]):
                    return is_missing or traverse_and_check_missing(row, node["Left"])
                else:
                    return is_missing or traverse_and_check_missing(row, node["Right"])
            return is_missing

        missing_reliance_count = dataset.rdd.map(
            lambda row: traverse_and_check_missing(row.asDict(), self.tree_)
        ).sum()
        return missing_reliance_count / dataset.count()

    def save(self, path):
        """Save the trained model."""
        with open(path, "w") as f:
            json.dump(self.tree_, f)

    def print_tree(self):
        """Pretty-print the decision tree."""

        def recurse(node, depth):
            indent = "  " * depth
            if "Predict" in node:
                print(f"{indent}Predict: {node['Predict']}")
            else:
                print(f"{indent}If ({node['If']})")
                recurse(node["Left"], depth + 1)
                print(f"{indent}Else")
                recurse(node["Right"], depth + 1)

        recurse(self.tree_, 0)


class PySparkMARFClassifier:
    """Missingness-avoiding random forest classifier."""

    def __init__(
        self,
        numTrees=100,
        maxDepth=10,
        alpha=1.0,
        bootstrap=True,
        seed=None,
        criterion="gini",
        labelCol="label",
    ):
        self.numTrees = numTrees
        self.maxDepth = maxDepth
        self.alpha = alpha
        self.bootstrap = bootstrap
        self.seed = seed
        self.labelCol = labelCol
        self.criterion = criterion
        self.trees = []

    def _fit(self, dataset):
        """Fit the Random Forest."""
        self.trees = []
        for i in range(self.numTrees):
            sample = dataset.sample(
                withReplacement=self.bootstrap, fraction=1.0, seed=self.seed
            )
            tree = PySparkMADTClassifier(
                maxDepth=self.maxDepth,
                criterion=self.criterion,
                alpha=self.alpha,
                labelCol=self.labelCol,
            )
            tree._fit(sample)
            self.trees.append(tree)

    def predict(self, dataset):
        """Predict using majority voting from all trees."""
        predictions = [tree.predict(dataset) for tree in self.trees]
        final_predictions = [
            max(set(preds), key=preds.count) for preds in zip(*predictions)
        ]
        return final_predictions

    def compute_missingness_reliance(self, dataset):
        """Compute average missingness reliance across all trees."""
        reliance = [tree.compute_missingness_reliance(dataset) for tree in self.trees]
        return sum(reliance) / len(reliance)

    def print_forest(self):
        """Print all decision trees in the forest."""
        for i, tree in enumerate(self.trees):
            print(f"Tree {i + 1}:")
            tree.print_tree()
            print("\n")

    def save(self, path):
        """Save the Random Forest model."""
        with open(path, "w") as f:
            json.dump([tree.tree_ for tree in self.trees], f)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("HistogramBasedDecisionTree").getOrCreate()

    # Load Dataset
    data_path = "./src/data/sample_data.csv"
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
        criterion="gini", maxDepth=3, alpha=0.1, labelCol=label
    )
    madt._fit(df)
    print("Trained MADT Tree:")
    madt.print_tree()

    # Predict
    predictions = madt.predict(df)
    print("Predictions:", predictions)

    # Compute Missingness Reliance
    reliance = madt.compute_missingness_reliance(df)
    print(f"Missingness reliance: {reliance:.4f}")

    # Save Model
    madt.save("./src/models/madt_model.json")

    # Train PySparkMARFClassifier
    marf = PySparkMARFClassifier(numTrees=3, maxDepth=3, alpha=0.1, labelCol=label)
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
