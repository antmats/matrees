import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array


def check_missingness_mask(M, X):
    if M is None:
        return None
    M = check_array(M)
    if not M.shape[0] == X.shape[0]:
        raise ValueError("The number of samples in X and M must be the same.")
    return M


def get_ensemble_missingness_reliance_from_df(df, X, M, equality_left=True):
    """Compute the missingness reliance proportion for an ensemble model given
    a DataFrame containing the structure of the trees within the ensemble.

    Parameters:
        df (DataFrame): DataFrame specifying the tree ensemble structure
        X (ndarray): Input data
        M (ndarray): Missingness mask
        equality_left (bool): Whether the left subtree contains values equal to the split threshold

    Returns:
        float: Missingness reliance proportion (rho_metric).
    """
    n_trees = df["Tree"].max() + 1
    children = dict(zip(df["ID"], zip(df["Yes"], df["No"])))
    features = dict(zip(df["ID"], df["Feature"]))
    thresholds = dict(zip(df["ID"], df["Split"]))

    n_samples, n_features = X.shape

    feat_id = {f"f{i}": i for i in range(n_features)}

    def check_missing(node, I, X, M):
        f = features[node]

        if f == "Leaf" or len(I) == 0:
            return []

        fid = feat_id.get(f, None)
        # TODO: Is this necessary?
        if fid is None:
            return []

        is_missing = M[I, fid].astype(bool)
        I_na = I[is_missing]
        I_o = I[~is_missing]

        left, right = children[node]
        if equality_left:
            I_l = I_o[X[I_o, fid] <= thresholds[node]]
            I_r = I_o[X[I_o, fid] > thresholds[node]]
        else:
            I_l = I_o[X[I_o, fid] < thresholds[node]]
            I_r = I_o[X[I_o, fid] >= thresholds[node]]

        miss_l = check_missing(left, I_l, X, M)
        miss_r = check_missing(right, I_r, X, M)

        return list(I_na) + miss_l + miss_r

    miss_rows = []
    for tree in range(n_trees):
        root_node = f"{tree}-0"
        miss_feat = check_missing(root_node, np.arange(n_samples), X, M)
        miss_rows.extend(miss_feat)

    rho_metric = len(np.unique(miss_rows)) / n_samples

    return rho_metric


def _tree_to_dataframe(tree, idx=0):
    n_nodes = tree.node_count
    tree_index = np.full(n_nodes, idx)
    node_ids = [f"{idx}-{i}" for i in range(n_nodes)]
    features = ["Leaf" if f == -2 else "f%d" % f for f in tree.feature]
    thresholds = tree.threshold
    children_left = [f"{idx}-{c}" % c if c > -1 else np.nan for c in tree.children_left]
    children_right = [
        f"{idx}-{c}" % c if c > -1 else np.nan for c in tree.children_right
    ]

    return pd.DataFrame(
        {
            "Tree": tree_index,
            "ID": node_ids,
            "Feature": features,
            "Split": thresholds,
            "Yes": children_left,
            "No": children_right,
        }
    )


def get_dt_missingness_reliance(dt, X, M):
    """Compute the fraction of rows in X for which a decision tree uses missing
    features.

    Parameters:
        dt (decision tree model): Model to evaluate
        X (ndarray or dataframe): Input data
        M (ndarray): Missingness mask

    Returns:
        float: The missingness reliance proportion
    """
    df = _tree_to_dataframe(dt.tree_)
    return get_ensemble_missingness_reliance_from_df(df, X, M)
