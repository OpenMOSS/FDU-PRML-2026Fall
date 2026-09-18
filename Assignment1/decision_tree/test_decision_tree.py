"""Decision tree test on the bundled UCI Wine dataset.

Run with::

    python test_decision_tree.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decision_tree import DecisionTreeClassifier
from viz_tree import plot_tree


def test_dt_classification():
    data_dir = Path(__file__).parent / "dataset"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "wine_train.csv"
    test_path = data_dir / "wine_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "wine_train.csv / wine_test.csv not found in the dataset directory."
        )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    feature_names = list(df_train.columns[:-1])
    X = df_train[feature_names].to_numpy(dtype=float)
    y = df_train["label"].to_numpy(dtype=int)
    X_test = df_test[feature_names].to_numpy(dtype=float)
    y_test = df_test["label"].to_numpy(dtype=int)

    criteria = ["info_gain", "info_gain_ratio", "gini", "error_rate"]
    min_acc = 0.85

    for crit in criteria:
        print(f"\n=== Criterion: {crit} ===")
        dt_clf = DecisionTreeClassifier(criterion=crit, random_state=0)
        dt_clf.fit(X, y)
        preds = dt_clf.predict(X_test)
        acc = (preds == y_test).mean()
        print(f"Accuracy: {acc:.4f}")
        print(f"Tree depth: {dt_clf.tree_depth}; leaves: {dt_clf.tree_leaf_num}")

        # Basic model-structure checks.  These do not constrain the exact
        # tree shape, since different valid implementations may choose
        # different but equivalent splits.
        if dt_clf.feature_importances_ is None:
            raise AssertionError("feature_importances_ was not created")
        if dt_clf.feature_importances_.shape != (len(feature_names),):
            raise AssertionError("feature_importances_ has an unexpected shape")
        if not np.isclose(dt_clf.feature_importances_.sum(), 1.0, atol=1e-8):
            raise AssertionError("feature_importances_ should sum to 1")
        if dt_clf.tree_depth < 1 or dt_clf.tree_leaf_num < 1:
            raise AssertionError("The fitted tree should contain at least one leaf")

        if acc < min_acc:
            raise AssertionError(
                f"Accuracy {acc:.4f} below threshold {min_acc} for criterion {crit}"
            )

        fig, ax = plot_tree(
            dt_clf,
            feat_names=feature_names,
            class_names=["0", "1", "2"],
            show_split_score=True,
            show_leaf_samples=True,
            top_padding=0.18,
        )
        ax.set_title(f"Decision Tree ({crit}) acc={acc:.2f}")
        fig.savefig(output_dir / f"wine_{crit}.png")
        plt.close(fig)


if __name__ == "__main__":
    test_dt_classification()
