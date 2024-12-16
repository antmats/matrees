import copy

import numpy as np
from sklearn.tree import _tree


def entropy(distr):
    n = sum(distr)
    ps = [n_i / n for n_i in distr]
    return -sum(p * np.log2(p) if p > 0 else 0 for p in ps)


def info_gain_scorer(n_low, low_distr, n_high, high_distr):
    return -(
        (n_low * entropy(low_distr) + n_high * entropy(high_distr)) 
        / (n_low + n_high)
    )


def gini_impurity(distr):
    n = sum(distr)
    ps = [n_i / n for n_i in distr]
    return 1 - sum(p**2 for p in ps)


def gini_scorer(n_low, low_distr, n_high, high_distr):
    return -(
        (n_low * gini_impurity(low_distr) + n_high * gini_impurity(high_distr))
        / (n_low + n_high)
    )


class TreeNode:
    def __init__(
        self,
        value,
        *,
        n_node_samples,
        weighted_n_node_samples,
        depth,
        feature=None,
        threshold=None,
        left_subtree=None,
        right_subtree=None,
        missingness_reliance=None,
    ):
        self.value = value
        self.n_node_samples = n_node_samples
        self.weighted_n_node_samples = weighted_n_node_samples
        self.depth = depth

        undefined = _tree.TREE_UNDEFINED
        self.feature = feature if feature is not None else undefined
        self.threshold = threshold if threshold is not None else undefined

        self.left_subtree = left_subtree
        self.right_subtree = right_subtree

        self.missingness_reliance = missingness_reliance

    def predict(self, x):
        if self.feature == _tree.TREE_UNDEFINED:
            return self.value

        if x[self.feature] <= self.threshold:
            return self.left_subtree.predict(x)
        else:
            return self.right_subtree.predict(x)

    @property
    def is_leaf(self):
        return self.feature == _tree.TREE_UNDEFINED

    def __eq__(self, other):
        return (
            isinstance(other, TreeNode)
            # np.array_equal works for both arrays and scalars.
            and np.array_equal(self.value, other.value)
        )


def count_nodes(node):
    if node is not None:
        return (
            1 + count_nodes(node.left_subtree) + count_nodes(node.right_subtree)
        )
    return 0


class SklearnTree:
    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        # Only single-output trees are supported.
        self.n_classes = np.array([n_classes])
        self.n_outputs = 1
        self.max_n_classes = n_classes

        self.nodes = []
        self.max_depth = None
        self.node_count = None
        self.capacity = None
        self.n_leaves = None

        self.children_left = []
        self.children_right = []
        self.feature = []
        self.threshold = []
        self.value = []
        #self.impurity = []
        self.missingness_reliance = []
        self.n_node_samples = []
        self.weighted_n_node_samples = []

    def populate(self, root):
        self._populate(root, index=0)
        self.max_depth = max(node.depth for node in self.nodes)
        self.node_count = len(self.nodes)
        self.capacity = len(self.nodes)
        self.n_leaves = sum(
            node.feature == _tree.TREE_UNDEFINED for node in self.nodes
        )
        self._to_arrays()
        return self

    def _populate(self, node, index):
        def append(node):
            self.nodes.append(node)
            self.children_left.append(node.left_child)
            self.children_right.append(node.right_child)
            self.feature.append(node.feature)
            self.threshold.append(node.threshold)
            self.value.append(node.value)
            self.missingness_reliance.append(node.missingness_reliance)
            self.n_node_samples.append(node.n_node_samples)
            self.weighted_n_node_samples.append(node.weighted_n_node_samples)

        node = copy.deepcopy(node)
        node.index = index

        if node.feature == _tree.TREE_UNDEFINED:
            node.left_child = _tree.TREE_LEAF
            node.right_child = _tree.TREE_LEAF

            append(node)
        else:
            left_idx = len(self.children_left) + 1
            right_idx = left_idx + count_nodes(node.left_subtree)

            node.left_child = left_idx
            node.right_child = right_idx

            append(node)

            self._populate(node.left_subtree, left_idx)
            self._populate(node.right_subtree, right_idx)

    def _to_arrays(self):
        self.nodes = np.array(self.nodes)
        self.children_left = np.array(self.children_left)
        self.children_right = np.array(self.children_right)
        self.feature = np.array(self.feature)
        self.threshold = np.array(self.threshold)
        self.value = np.array(self.value).reshape(-1, 1, self.max_n_classes)
        self.n_node_samples = np.array(self.n_node_samples)
        self.weighted_n_node_samples = np.array(self.weighted_n_node_samples)

    def apply(self, X):
        out = []
        n_samples = X.shape[0]
        for i in range(n_samples):
            node = self.nodes[0]  # Start at the root
            while node.left_child != _tree.TREE_LEAF:
                if X[i, node.feature] <= node.threshold:
                    node = self.nodes[node.left_child]
                else:
                    node = self.nodes[node.right_child]
            out.append(node.index)
        return np.array(out)


def convert_to_sklearn_tree(root_node, n_features, n_classes):
    tree = SklearnTree(n_features, n_classes).populate(root_node)
    return tree
