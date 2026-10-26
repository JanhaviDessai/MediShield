
"""Evaluation metrics for Medical AutoML classification models."""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true,
    y_pred,
    class_names: list[str] | None = None,
    y_prob=None,
) -> dict:
    """Calculate classification metrics and diagnostic tables."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty test set.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Actual labels and predictions must have equal lengths."
        )

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    if class_names is not None and len(class_names) > max(labels):
        display_names = [
            str(class_names[label]) for label in labels
        ]
    else:
        display_names = [str(label) for label in labels]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.index.name = "class"

    confusion_df = pd.DataFrame(
        matrix,
        index=[f"Actual: {name}" for name in display_names],
        columns=[f"Predicted: {name}" for name in display_names],
    )

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    # ROC AUC is only included when valid probability estimates are supplied.
    if y_prob is not None:
        probabilities = np.asarray(y_prob)

        try:
            if len(labels) == 2 and probabilities.ndim == 2:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true,
                        probabilities[:, 1],
                    )
                )
            elif len(labels) > 2 and probabilities.ndim == 2:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true,
                        probabilities,
                        multi_class="ovr",
                        average="macro",
                        labels=labels,
                    )
                )
        except (ValueError, IndexError):
            metrics["roc_auc"] = None

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_df,
        "classification_report": report_df,
    }
