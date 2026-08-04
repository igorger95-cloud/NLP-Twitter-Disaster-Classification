from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Accuracy, Precision, Recall and F1-score.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
