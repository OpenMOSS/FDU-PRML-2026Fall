import numpy as np


def accuracy_score(y_true, y_pred):
    """Compute the classification accuracy for one-dimensional label arrays.

    Each sample has one ground-truth label and one predicted label.

    Parameters
    ----------
    y_true : 1d array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : 1d array-like of shape (n_samples,)
        Predicted class labels, in the same sample order as y_true.

    Returns
    -------
    accuracy : float
        Classification accuracy in [0, 1]. Higher values indicate better
        predictions.

    Notes
    -----
    The two input arrays must have the same length.

    Examples
    --------
    >>> accuracy_score([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    # =============== TODO (students) ===============

    # ===============================================
    raise NotImplementedError("Implement accuracy_score")


def mean_squared_error(y_true, y_pred):
    """Compute the mean squared error for one-dimensional numeric arrays.

    Mean squared error (MSE) measures numeric prediction error. A smaller
    value indicates a closer match, and a value of zero means all
    predictions are exact.

    Parameters
    ----------
    y_true : 1d array-like of shape (n_samples,)
        Ground-truth numeric values.
    y_pred : 1d array-like of shape (n_samples,)
        Predicted numeric values, in the same sample order as y_true.

    Returns
    -------
    mse : float
        Non-negative mean squared prediction error.

    Notes
    -----
    The two input arrays must have the same length. MSE is intended for
    numeric prediction errors; it is not a general classification metric
    for arbitrary class labels.

    Examples
    --------
    >>> mean_squared_error([0.0, 1.0, 2.0], [0.0, 2.0, 1.0])
    0.6666666666666666
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    # =============== TODO (students) ===============

    # ===============================================
    raise NotImplementedError("Implement mean_squared_error")
