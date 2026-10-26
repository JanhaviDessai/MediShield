
"""Leakage-aware preprocessing for Medical AutoML."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features from the target labels."""
    if df.empty:
        raise ValueError("The dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    if df.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")

    target = df[target_column].copy()
    features = df.drop(columns=[target_column]).copy()

    # Rows without known labels cannot be used for supervised training.
    valid_target = target.notna()

    if not valid_target.any():
        raise ValueError(
            "The target column contains no observed labels."
        )

    features = features.loc[valid_target].reset_index(drop=True)
    target = target.loc[valid_target].reset_index(drop=True)

    # Avoid accidental training on a constant or missing target.
    if target.nunique(dropna=True) < 2:
        raise ValueError(
            "The target must contain at least two distinct labels."
        )

    if features.shape[1] == 0:
        raise ValueError(
            "No input features remain after removing the target column."
        )

    return features, target


def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build numeric and categorical preprocessing pipelines."""
    if X.empty or X.shape[1] == 0:
        raise ValueError("Input features are empty.")

    if X.columns.duplicated().any():
        raise ValueError("Input features contain duplicate column names.")

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    transformers: list[tuple[str, Any, list]] = []

    if numeric_columns:
        numeric_steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            )
        ]

        if scale_numeric:
            numeric_steps.append(
                ("scaler", StandardScaler())
            )

        numeric_pipeline = Pipeline(steps=numeric_steps)

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No supported input columns were found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_feature_names(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """Return the names of transformed features."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "Feature names are unavailable. Fit the preprocessor first."
        ) from exc
