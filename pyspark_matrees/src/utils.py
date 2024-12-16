from pyspark.ml.classification import Classifier
from pyspark.sql.functions import col, lit, when, mean
import numpy as np
import json

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt


# def convert_json_to_sklearn(json_tree, features):
#     """
#     Convert a JSON representation of a decision tree into a scikit-learn DecisionTreeClassifier.

#     Args:
#         json_tree (dict): The JSON representation of the decision tree.
#         features (list): List of feature names.

#     Returns:
#         DecisionTreeClassifier: A scikit-learn DecisionTreeClassifier object.
#     """

#     def traverse_json(node, depth=0, left_child=None, right_child=None):
#         """
#         Recursively build a tree-compatible structure.
#         """
#         if "Predict" in node:
#             # Leaf node
#             return {"value": node["Predict"], "left": None, "right": None}
#         else:
#             # Split node
#             feature, threshold = node["If"].split(" <= ")
#             left = traverse_json(node["Left"], depth + 1)
#             right = traverse_json(node["Right"], depth + 1)
#             return {
#                 "feature": feature.strip(),
#                 "threshold": float(threshold.strip()),
#                 "left": left,
#                 "right": right,
#             }

#     def build_sklearn_tree(node, depth=0):
#         """
#         Build arrays compatible with scikit-learn DecisionTreeClassifier.
#         """
#         # Recursively extract node structure
#         tree_structure = []

#         def recursive_build(node, parent_idx, is_left):
#             node_idx = len(tree_structure)
#             tree_structure.append(
#                 {
#                     "node_idx": node_idx,
#                     "parent_idx": parent_idx,
#                     "is_left": is_left,
#                     "feature": node.get("feature", None),
#                     "threshold": node.get("threshold", None),
#                     "value": node.get("value", None),
#                 }
#             )
#             if node["left"]:
#                 recursive_build(node["left"], node_idx, True)
#             if node["right"]:
#                 recursive_build(node["right"], node_idx, False)

#         # Start building tree structure
#         recursive_build(node, parent_idx=-1, is_left=False)

#         # Create arrays for scikit-learn
#         n_nodes = len(tree_structure)
#         children_left = np.zeros(n_nodes, dtype=int) - 1
#         children_right = np.zeros(n_nodes, dtype=int) - 1
#         feature = np.zeros(n_nodes, dtype=int) - 2  # -2 means unused feature
#         threshold = np.zeros(n_nodes, dtype=float) - 2.0
#         values = []

#         for node in tree_structure:
#             idx = node["node_idx"]
#             if node["feature"] is not None:
#                 feature[idx] = features.index(node["feature"])
#                 threshold[idx] = node["threshold"]
#             if node["value"] is not None:
#                 values.append(node["value"])
#             else:
#                 values.append(None)

#         # Build tree object
#         tree = _tree.Tree(n_nodes, n_features=len(features), dtype=np.float64)
#         tree.children_left[:] = children_left
#         tree.children_right[:] = children_right
#         tree.feature[:] = feature
#         tree.threshold[:] = threshold

#         return tree

#     # Parse JSON to a compatible structure
#     parsed_tree = traverse_json(json_tree)
#     return build_sklearn_tree(parsed_tree)


def partition_and_save_csv(df, output_path, num_partitions=8):
    """
    Partition the given DataFrame into multiple parts and save as CSV files.

    Args:
        df (DataFrame): The Spark DataFrame to partition.
        output_path (str): The path to save the partitioned CSV files.
        num_partitions (int): Number of partitions (default is 8).
    """
    # Repartition the DataFrame
    df = df.repartition(num_partitions)

    # Write DataFrame to the output path as CSV
    df.write.option("header", True).csv(output_path)
