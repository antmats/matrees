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
        # Stop if we've hit max depth or have no rows left
        if depth >= self.maxDepth or dataset.count() == 0:
            return self._get_leaf(dataset)

        # Check if all samples have the same label
        label_counts = (
            dataset.groupBy(self.labelCol)
            .agg(spark_sum("sample_weight").alias("weight"))
            .collect()
        )
        if len(label_counts) == 1:
            return {"Predict": label_counts[0][self.labelCol]}

        # Find the best split using histogram bins for each feature
        best_feature, best_threshold, best_score = None, None, float("-inf")
        for feature in dataset.columns:
            if feature not in [
                self.labelCol,
                "sample_weight",
            ] and not feature.startswith("missing_"):
                score, threshold = self._best_split(dataset, feature)
                if score is not None and score > best_score:
                    best_feature, best_threshold, best_score = feature, threshold, score

        # If no split improves the score, return a leaf
        if best_feature is None:
            return self._get_leaf(dataset)

        # Otherwise, split the dataset
        left_data = dataset.filter(col(best_feature) <= best_threshold)
        right_data = dataset.filter(col(best_feature) > best_threshold)

        return {
            "If": f"{best_feature} <= {best_threshold}",
            "Left": self._build_tree(left_data, depth + 1),
            "Else": f"{best_feature} > {best_threshold}",
            "Right": self._build_tree(right_data, depth + 1),
        }

    def _best_split(self, dataset, feature):
        """
        Find the best split for a feature using histogram bins, including
        a missingness penalty proportional to the fraction of samples
        at this node that are missing this feature.
        """
        # Compute node-level missing penalty
        node_weight = dataset.agg(spark_sum("sample_weight")).collect()[0][0] or 0.0
        if node_weight == 0:
            return None, None

        missing_count = (
            dataset.filter(col(f"missing_{feature}") == 1)
            .agg(spark_sum("sample_weight"))
            .collect()[0][0]
            or 0.0
        )
        # The fraction of this node that is missing the current feature
        missing_frac = missing_count / node_weight

        # We multiply by alpha to scale how strongly we penalize missingness
        node_missing_penalty = missing_frac * self.alpha

        # Build histogram bins (unique values sorted)
        histogram = (
            dataset.groupBy(feature)
            .agg(
                spark_sum("sample_weight").alias("total_weight"),
                spark_sum(
                    when(col(self.labelCol) == 1, col("sample_weight")).otherwise(0)
                ).alias("positive_weight"),
            )
            .orderBy(asc(feature))
            .collect()
        )

        # If we don't have at least 2 distinct values, we can't split
        if len(histogram) < 2:
            return None, None

        total_weight = sum(row["total_weight"] for row in histogram)
        total_positive = sum(row["positive_weight"] for row in histogram)

        weight_left, positive_left = 0.0, 0.0
        best_score, best_threshold = float("-inf"), None

        # Iterate through histogram bins to find the best threshold
        for i in range(len(histogram) - 1):
            weight_left += histogram[i]["total_weight"]
            positive_left += histogram[i]["positive_weight"]

            weight_right = total_weight - weight_left
            positive_right = total_positive - positive_left

            if weight_left == 0 or weight_right == 0:
                continue

            # Calculate Gini for left and right
            gini_left = gini_impurity(positive_left, weight_left)
            gini_right = gini_impurity(positive_right, weight_right)

            # Weighted Gini
            weighted_gini = (weight_left / total_weight) * gini_left + (
                weight_right / total_weight
            ) * gini_right

            # We define a "score" = -(weighted_gini) - missing penalty
            # We want to MAXIMIZE this score, i.e. minimize Gini + penalty
            gini_score = -weighted_gini - node_missing_penalty

            # If this is the best so far, record it
            if gini_score > best_score:
                best_score = gini_score
                # The threshold is the midpoint between this bin's value and the next bin's value
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
            feature, _, threshold_str = node["If"].split()
            threshold = float(threshold_str)
            if row[feature] <= threshold:
                return traverse_tree(row, node["Left"])
            else:
                return traverse_tree(row, node["Right"])

        return dataset.rdd.map(
            lambda row: traverse_tree(row.asDict(), self.tree_)
        ).collect()

    def compute_missingness_reliance(self, dataset):
        """
        Compute the proportion of rows that traverse a path involving a missing feature
        (i.e., how often missing_ col influences the path). This is just one example
        of a 'missingness reliance' metric.
        """

        def traverse_and_check_missing(row, node):
            # If we're at a leaf, no further checks
            if "Predict" in node:
                return 0
            feature = node["If"].split()[0]
            is_missing = row.get(f"missing_{feature}", 0)
            if row[feature] is not None:
                threshold = float(node["If"].split()[2])
                if row[feature] <= threshold:
                    return is_missing or traverse_and_check_missing(row, node["Left"])
                else:
                    return is_missing or traverse_and_check_missing(row, node["Right"])
            # If the value is truly None (somehow) and we missed it, we treat that as missing
            return is_missing

        missing_reliance_count = dataset.rdd.map(
            lambda row: traverse_and_check_missing(row.asDict(), self.tree_)
        ).sum()

        total_count = dataset.count()
        if total_count == 0:
            return 0.0
        return missing_reliance_count / total_count

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
        """Fit the Random Forest by training multiple MADT trees on bootstrap samples."""
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
        # Each tree returns a list of predictions
        predictions_per_tree = [tree.predict(dataset) for tree in self.trees]
        # Transpose to get predictions for each row across all trees
        final_predictions = []
        for row_preds in zip(*predictions_per_tree):
            # Majority vote
            pred = max(set(row_preds), key=row_preds.count)
            final_predictions.append(pred)
        return final_predictions

    def compute_missingness_reliance(self, dataset):
        """Compute average missingness reliance across all trees."""
        if not self.trees:
            return 0.0
        reliances = [tree.compute_missingness_reliance(dataset) for tree in self.trees]
        return sum(reliances) / len(reliances)

    def print_forest(self):
        """Print all decision trees in the forest."""
        for i, tree in enumerate(self.trees):
            print(f"Tree {i + 1}:")
            tree.print_tree()
            print()

    def save(self, path):
        """Save the Random Forest model as JSON."""
        with open(path, "w") as f:
            json.dump([tree.tree_ for tree in self.trees], f)


if __name__ == "__main__":
    spark = SparkSession.builder.appName(
        "MissingnessAvoidingDecisionTree"
    ).getOrCreate()

    # Load Dataset
    data_path = "./src/data/sample_data.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Make sure label is numeric, e.g., 0/1.
    # If 'credit_approved' column is string "yes"/"no", map it to 1/0 first.
    # df = df.withColumn("credit_approved", when(col("credit_approved") == "yes", 1).otherwise(0))

    # Define features and label
    features = ["credit_age", "income", "full_time", "part_time", "student"]
    label = "credit_approved"

    # Add Missingness Mask and Impute Missing Values
    for feature in features:
        # Create a missing_ column
        df = df.withColumn(
            f"missing_{feature}", when(col(feature).isNull(), 1.0).otherwise(0.0)
        )
        # Impute with mean
        mean_value = df.select(mean(col(feature))).first()[0]
        df = df.withColumn(
            feature,
            when(col(feature).isNull(), lit(mean_value)).otherwise(col(feature)),
        )

    # Split into train/test to avoid overfitting:
    # train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    # For brevity, we’ll just use the entire df as training here.

    # Train a single decision tree
    madt = PySparkMADTClassifier(
        criterion="gini",
        maxDepth=3,
        alpha=0.1,  # Lower alpha if you still find the penalty too strong
        labelCol=label,
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

    # Save single-tree model
    madt.save("./src/models/madt_model.json")

    # Train a random forest of 3 trees
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

    # Save the entire forest
    marf.save("./src/models/marf_model")

    spark.stop()
