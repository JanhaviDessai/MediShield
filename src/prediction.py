
"""Batch prediction using saved disease-specific model packages."""

import pandas as pd

from src.model_training import load_model_package, IDENTIFIER_COLUMNS


def prepare_prediction_input(
    input_data: pd.DataFrame,
    package: dict,
) -> pd.DataFrame:
    """Validate prediction features and order them as used in training."""
    if input_data.empty:
        raise ValueError("The uploaded CSV contains no records.")

    if input_data.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")

    feature_columns = package["feature_columns"]
    ignored_ids = [
        column for column in input_data.columns
        if str(column).strip().lower() in IDENTIFIER_COLUMNS
        and column not in feature_columns
    ]

    usable = input_data.drop(columns=ignored_ids, errors="ignore")

    missing = [
        column for column in feature_columns
        if column not in usable.columns
    ]

    if missing:
        raise ValueError(
            "The CSV is missing required model features: "
            + ", ".join(missing)
        )

    # Extra non-ID columns are rejected to prevent silent schema mistakes.
    extra = [
        column for column in usable.columns
        if column not in feature_columns
    ]
    if extra:
        raise ValueError(
            "Unexpected columns in the CSV: "
            + ", ".join(map(str, extra))
        )

    return usable.loc[:, feature_columns].copy()


def predict_records(
    input_data: pd.DataFrame,
    model_slug: str,
    package: dict | None = None,
) -> pd.DataFrame:
    """Predict labels and probabilities for a CSV's records."""
    if package is None:
        package = load_model_package(model_slug)

    features = prepare_prediction_input(input_data, package)
    pipeline = package["pipeline"]

    try:
        encoded_predictions = pipeline.predict(features)
    except Exception as exc:
        raise ValueError(
            f"Prediction failed. Check the CSV feature values: {exc}"
        ) from exc

    class_names = package["class_names"]
    labels = []

    for prediction in encoded_predictions:
        index = int(prediction)
        if not 0 <= index < len(class_names):
            raise ValueError("The model returned an unknown class.")
        labels.append(class_names[index])

    # Preserve the patient's ID in the results when one was supplied.
    result = pd.DataFrame(index=input_data.index)
    id_columns = [
        column for column in input_data.columns
        if str(column).strip().lower() in IDENTIFIER_COLUMNS
    ]

    for column in id_columns:
        result[column] = input_data[column]

    result["predicted_label"] = labels

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        model_classes = pipeline.named_steps["model"].classes_

        for index, encoded_class in enumerate(model_classes):
            class_index = int(encoded_class)
            class_name = str(class_names[class_index])
            result[f"probability_{class_name}"] = probabilities[:, index]

    result.insert(0, "disease_model", package["disease_name"])
    return result.reset_index(drop=True)
