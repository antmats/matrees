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
