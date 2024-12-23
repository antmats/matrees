import json
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from warnings import warn

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, sum as spark_sum, mean, asc
import json


from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    is_classifier,
    _fit_context,
)
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    _check_sample_weight,
)

from .tree import (
    info_gain_scorer,
    gini_scorer,
    TreeNode,
    convert_to_sklearn_tree,
)

CRITERIA_CLF = {"info_gain": info_gain_scorer, "gini": gini_scorer}


def check_missingness_mask(M, X, allow_none=True):
    if M is None and allow_none:
        return None
    M = check_array(M)
    if not M.shape[0] == X.shape[0]:
        raise ValueError("The number of samples in X and M must be the same.")
    return M


class BaseMADT(BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: dict = {
        "max_depth": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "alpha": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(self, max_depth, random_state, alpha):
        super().__init__()
        self.max_depth = max_depth
        self.random_state = random_state
        self.alpha = alpha

    def _fit(
        self,
        X,
        y,
        M=None,
        *,
        clear_X_M=True,
        check_input=True,
        sample_weight=None,
        missing_values_in_feature_mask=None,
    ):
        if check_input:
            X, y = check_X_y(X, y)

        # Meta-estimators may pass labels as a column vector, but we need them
        # as a 1-D array.
        if y.ndim > 1:
            y = y.ravel()

        if is_classifier(self):
            labels = unique_labels(y)
            n_labels = len(labels)
            if not np.all(labels == np.arange(n_labels)):
                raise ValueError(
                    "Labels are expected to be in the range 0 to n_classes-1."
                )

        if missing_values_in_feature_mask is not None:
            if M is not None:
                warn(
                    "Both `M` and `missing_values_in_feature_mask` are "
                    "provided. `M` will be used."
                )
            else:
                M = missing_values_in_feature_mask

        M = check_missingness_mask(M, X)

        sample_weight = _check_sample_weight(sample_weight, X)

        random_state = check_random_state(self.random_state)

        self.n_features_in_ = X.shape[1]
        if is_classifier(self):
            self.classes_ = unique_labels(y)
            self.n_classes_ = len(self.classes_)

        self._split_features = []
        self.root_ = self._make_tree(X, y, M, sample_weight, 0, random_state)

        if M is not None:
            self.root_.X, self.root_.M = X, M
            self._compute_missingness_reliance_for_all_nodes(self.root_, clear_X_M)

        n_classes = self.n_classes_ if is_classifier(self) else 1
        self.tree_ = convert_to_sklearn_tree(self.root_, self.n_features_in_, n_classes)

        return self

    def _validate_X_predict(self, X, check_input=True):
        if check_input:
            X = check_array(X)
        n_features = X.shape[1]
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.n_features_in_} "
                "features were expected."
            )
        return X

    def _support_missing_values(self, X):
        return False

    def _compute_missing_values_in_feature_mask(self, X, estimator_name=None):
        return None

    def predict(self, X, check_input=True, return_proba=False):
        check_is_fitted(self)
        if check_input:
            X = self._validate_X_predict(X)
        yp = np.array([self._predict_one(x) for x in X])
        if is_classifier(self) and not return_proba:
            # Classes are ordered from 0 to n_classes-1.
            return np.argmax(yp, axis=1)
        return yp

    def _predict_one(self, x):
        return self.root_.predict(x)

    def _make_tree(self, X, y, M, sample_weight, depth, random_state):
        """Recursively build the decision tree."""

        node_value = self._get_node_value(y, sample_weight)
        n_node_samples = len(X)
        weighted_n_node_samples = sum(sample_weight)

        # Check if the training set is completely homogeneous, or if the
        # maximum depth has been reached.
        if depth >= self.max_depth or self._is_homogeneous(
            node_value, y, sample_weight
        ):
            return TreeNode(
                node_value,
                n_node_samples=n_node_samples,
                weighted_n_node_samples=weighted_n_node_samples,
                depth=depth,
            )

        # Select the best feature to split on.
        n_features = X.shape[1]
        features = np.arange(n_features)
        random_state.shuffle(features)
        _split_score, split_feature, split_threshold = max(
            self._get_split_candidates(X, y, M, sample_weight, depth, random_state),
        )
        self._split_features.append(split_feature)

        if split_feature is None:
            return TreeNode(
                node_value,
                n_node_samples=n_node_samples,
                weighted_n_node_samples=weighted_n_node_samples,
                depth=depth,
            )

        # Split the training set into two parts and build the subtrees.

        data_left, data_right = self._split_by_feature(
            X, y, M, sample_weight, split_feature, split_threshold
        )

        left_subtree = self._make_tree(*data_left, depth + 1, random_state)
        right_subtree = self._make_tree(*data_right, depth + 1, random_state)

        if M is not None:
            # Store the training data and the missingness mask in the nodes to
            # compute the missingness reliance.
            left_subtree.X, left_subtree.M = data_left[0], data_left[2]
            right_subtree.X, right_subtree.M = data_right[0], data_right[2]

        if left_subtree == right_subtree:
            return left_subtree

        return TreeNode(
            node_value,
            n_node_samples=n_node_samples,
            weighted_n_node_samples=weighted_n_node_samples,
            feature=split_feature,
            threshold=split_threshold,
            left_subtree=left_subtree,
            right_subtree=right_subtree,
            depth=depth,
        )

    def _split_by_feature(self, X, y, M, sw, feature, threshold):
        left = X[:, feature] <= threshold
        right = ~left
        Xl, Xr = X[left], X[right]
        yl, yr = y[left], y[right]
        Ml, Mr = (M[left], M[right]) if M is not None else (None, None)
        swl = sw[left]
        swr = sw[right]
        return (Xl, yl, Ml, swl), (Xr, yr, Mr, swr)

    def _get_features_along_decision_path(self, x, max_depth=None):
        """Get the features along the decision path for the input `x`."""
        node = self.root_
        if max_depth is None:
            max_depth = self.max_depth
        while node.depth <= max_depth and node.left_subtree is not None:
            yield node.feature
            if x[node.feature] <= node.threshold:
                node = node.left_subtree
            else:
                node = node.right_subtree

    def _compute_missingness_reliance_for_all_nodes(self, node, clear_X_M=True):
        if node is None:
            return
        node.missingness_reliance = self.compute_missingness_reliance(
            node.X,
            node.M,
            max_depth=node.depth,
            check_input=False,
        )
        if clear_X_M:
            del node.X
            del node.M
        self._compute_missingness_reliance_for_all_nodes(node.left_subtree, clear_X_M)
        self._compute_missingness_reliance_for_all_nodes(node.right_subtree, clear_X_M)

    def compute_missingness_reliance(
        self,
        X,
        M,
        sample_mask=None,
        reduce=True,
        max_depth=None,
        check_input=True,
    ):
        check_is_fitted(self)
        if check_input:
            X = self._validate_X_predict(X)
            M = check_missingness_mask(M, X, allow_none=False)
        miss_reliance = np.zeros_like(M)
        for i, (x, m) in enumerate(zip(X, M)):
            if sample_mask is not None and not sample_mask[i]:
                continue
            for feature in self._get_features_along_decision_path(x, max_depth):
                miss_reliance[i, feature] = m[feature]
        if reduce:
            return np.mean(np.any(miss_reliance, axis=1))
        else:
            return miss_reliance

    def _get_split_candidates(self, X, y, M, sample_weight, depth, random_state):
        n_features = X.shape[1]
        features = np.arange(n_features)
        random_state.shuffle(features)
        for feature in features:
            yield self._best_split(X, y, feature, sample_weight=sample_weight, M=M)

    @abstractmethod
    def _get_node_value(self, y, sample_weight):
        """Get the value of a node in the decision tree."""

    @abstractmethod
    def _is_homogeneous(self, node_value, y, sample_weight):
        """Determine whether a node is homogeneous."""

    @abstractmethod
    def _best_split(self, X, y, feature, sample_weight, M=None):
        """Find the best split for a given feature."""


class MADTClassifier(ClassifierMixin, BaseMADT):
    """Missingness-avoiding decision tree classifier."""

    _parameter_constraints: dict = {
        **BaseMADT._parameter_constraints,
        "criterion": [StrOptions({"info_gain", "gini"})],
    }

    def __init__(self, criterion="gini", max_depth=3, random_state=None, alpha=1.0):
        super().__init__(max_depth, random_state, alpha)
        self.criterion = criterion

    def _get_node_value(self, y, sample_weight):
        node_value = np.zeros(self.n_classes_, dtype=float)
        for class_label, weight in zip(y, sample_weight):
            node_value[class_label] += weight
        node_value /= sum(sample_weight)
        return node_value

    def _is_homogeneous(self, node_value, y, sample_weight):
        return max(node_value) == 1.0

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, M=None, sample_weight=None):
        return super()._fit(X, y, M, sample_weight=sample_weight)

    def predict_proba(self, X, check_input=True):
        return self.predict(X, check_input, return_proba=True)

    def _best_split(self, X, y, feature, sample_weight, M=None):
        sorted_indices = np.argsort(X[:, feature])
        X_sorted = X[sorted_indices, feature]
        y_sorted = y[sorted_indices]
        sample_weight_sorted = sample_weight[sorted_indices]

        low_distr = np.zeros(self.n_classes_, dtype=float)
        high_distr = np.zeros(self.n_classes_, dtype=float)

        for class_label, weight in zip(y_sorted, sample_weight_sorted):
            high_distr[class_label] += weight

        n_low = 0
        n_high = np.sum(sample_weight)

        max_score = -np.inf
        i_max_score = None

        n = len(y)
        for i in range(n - 1):
            yi = y_sorted[i]
            wi = sample_weight_sorted[i]

            low_distr[yi] += wi
            high_distr[yi] -= wi

            n_low += wi
            n_high -= wi

            if n_low == 0:
                continue

            if n_high == 0:
                break

            # If the input is equal to the input at the next position, we will
            # not consider a split here.
            if np.isclose(X_sorted[i], X_sorted[i + 1]):
                continue

            criterion_function = CRITERIA_CLF[self.criterion]
            score = criterion_function(n_low, low_distr, n_high, high_distr)

            if score > max_score:
                max_score = score
                i_max_score = i

        if i_max_score is None:
            return -np.inf, None, None

        if M is not None:
            max_score -= self.alpha * np.mean(sample_weight * M[:, feature])

        split_threshold = 0.5 * (X_sorted[i_max_score] + X_sorted[i_max_score + 1])
        return max_score, feature, split_threshold


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
