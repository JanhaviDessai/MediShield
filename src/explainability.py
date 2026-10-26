
"""SHAP explanations for saved disease models."""

import numpy as np
import pandas as pd
import shap


def explain_records(
    package: dict,
    input_data: pd.DataFrame,
    max_rows: int = 20,
) -> dict:
    """Explain predictions for the supplied records."""
    if input_data.empty:
        raise ValueError("No records are available to explain.")

    pipeline = package["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    features = input_data.loc[:, package["feature_columns"]].head(max_rows)
    transformed = preprocessor.transform(features)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=float)
    background = np.asarray(
        package["background_transformed"],
        dtype=float,
    )
    feature_names = package["transformed_feature_names"]

    # SHAP explains the estimator after preprocessing.
    explainer = shap.Explainer(
        model.predict_proba,
        background,
        feature_names=feature_names,
    )
    shap_values = explainer(transformed)

    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "rows": len(features),
        "class_names": package["class_names"],
    }


def global_importance(explanation: dict) -> pd.DataFrame:
    """Rank transformed features by mean absolute SHAP value."""
    values = np.asarray(explanation["shap_values"].values)

    if values.ndim == 3:
        importance = np.mean(np.abs(values), axis=(0, 2))
    elif values.ndim == 2:
        importance = np.mean(np.abs(values), axis=0)
    else:
        raise ValueError(
            f"Unsupported SHAP output shape: {values.shape}"
        )

    return (
        pd.DataFrame(
            {
                "feature": explanation["feature_names"],
                "mean_absolute_shap": importance,
            }
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )


def individual_importance(
    explanation: dict,
    row_index: int,
    class_index: int,
) -> pd.DataFrame:
    """Return feature contributions for one record and class."""
    values = np.asarray(explanation["shap_values"].values)
    data = np.asarray(explanation["shap_values"].data)

    if not 0 <= row_index < values.shape[0]:
        raise IndexError("Record index is out of range.")

    row_values = values[row_index]

    if row_values.ndim == 2:
        if not 0 <= class_index < row_values.shape[1]:
            raise IndexError("Class index is out of range.")
        row_values = row_values[:, class_index]

    row_data = data[row_index]

    result = pd.DataFrame(
        {
            "feature": explanation["feature_names"],
            "transformed_value": row_data,
            "shap_value": row_values,
        }
    )
    result["absolute_shap"] = result["shap_value"].abs()

    return result.sort_values(
        "absolute_shap",
        ascending=False,
    ).reset_index(drop=True)
