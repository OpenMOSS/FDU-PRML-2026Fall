import numpy as np

EPS = 1e-12  # 避免分母为零


def _binary_counts(y_true, y_pred):
    """Count TP, FP, FN, and TN for binary labels {0, 1}.

    Label 1 is the positive class, and label 0 is the negative class.
    The two input arrays must have the same length and sample order.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (must be 0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (must be 0 or 1).

    Returns
    -------
    tp : int
        True positives (y_true == 1 and y_pred == 1).
    fp : int
        False positives (y_true == 0 and y_pred == 1).
    fn : int
        False negatives (y_true == 1 and y_pred == 0).
    tn : int
        True negatives (y_true == 0 and y_pred == 0).
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if not np.all(np.isin(y_true, (0, 1))) or not np.all(np.isin(y_pred, (0, 1))):
        raise ValueError("This template expects binary labels {0, 1}.")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def precision_score(y_true, y_pred):
    """Compute precision for the positive class (label 1).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels, in the same sample order as y_true.

    Returns
    -------
    precision : float
        Positive-class precision in [0, 1].

    Notes
    -----
    Use EPS to avoid division by zero when computing the score.
    """
    tp, fp, _, _ = _binary_counts(y_true, y_pred)

    # =============== TODO (students) ===============

    # ===============================================
    raise NotImplementedError("Implement precision_score")


def recall_score(y_true, y_pred):
    """Compute recall for the positive class (label 1).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels, in the same sample order as y_true.

    Returns
    -------
    recall : float
        Positive-class recall in [0, 1].

    Notes
    -----
    Use EPS to avoid division by zero when computing the score.
    """
    tp, _, fn, _ = _binary_counts(y_true, y_pred)

    # =============== TODO (students) ===============

    # ===============================================
    raise NotImplementedError("Implement recall_score")


def f1_score(y_true, y_pred):
    """Compute the F1 score for the positive class (label 1).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels, in the same sample order as y_true.

    Returns
    -------
    f1 : float
        Positive-class F1 score in [0, 1].

    Notes
    -----
    Use EPS to avoid division by zero when computing the score.
    """
    # =============== TODO (students) ===============

    # ===============================================
    raise NotImplementedError("Implement f1_score")
