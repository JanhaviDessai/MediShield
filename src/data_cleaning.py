

"""Data cleaning and data-quality reporting for Medical AutoML."""

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


MISSING_MARKERS = {
    "",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "nil",
    "?",
    "missing",
}

IDENTIFIER_NAMES = {
    "id",
    "patient_id",
    "patientid",
    "record_id",
    "recordid",
}

REPORT_COLUMNS = [
    "column",
    "action",
    "affected_cells",
]


def _is_text_dtype(series: pd.Series) -> bool:
    """Return whether a column contains string-like data."""
    return (
        pd.api.types.is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.StringDtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _normalise_text(series: pd.Series) -> pd.Series:
    """Trim whitespace and normalize recognized missing markers."""
    result = series.copy()

    # pandas StringDtype supports missing values and avoids row-wise
    # Python lambdas for most string-cleaning operations.
    if not isinstance(result.dtype, pd.StringDtype):
        result = result.astype("string")

    result = result.str.strip()

    marker_mask = (
        result.str.lower()
        .isin(MISSING_MARKERS)
        .fillna(False)
    )

    result = result.mask(marker_mask, pd.NA)

    return result


def clean_dataset(
    df: pd.DataFrame,
    *,
    normalize_numeric: bool = True,
    normalize_categories: bool = True,
    report_outliers: bool = True,
    outlier_iqr_multiplier: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean safe formatting issues and generate a data-quality report.

    This function does not:
    - Remove duplicate records automatically.
    - Remove outliers automatically.
    - Impute missing values.
    - Encode categorical features.
    - Scale numerical features.
    - Modify the target column differently from other columns.

    These transformations should be configured in the training pipeline
    to prevent data leakage and preserve consistent inference behavior.

    Returns:
        cleaned_df: Cleaned copy of the input dataset.
        report_df: Data-quality changes and warnings.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The uploaded dataset contains no rows.")

    if len(df.columns) == 0:
        raise ValueError("The uploaded dataset contains no columns.")

    if not isinstance(outlier_iqr_multiplier, (int, float)):
        raise TypeError("The outlier multiplier must be numeric.")

    if outlier_iqr_multiplier <= 0:
        raise ValueError("The outlier multiplier must be positive.")

    cleaned = df.copy(deep=True)
    report = []

    def record(column, action, affected=0):
        report.append({
            "column": str(column),
            "action": str(action),
            "affected_cells": int(affected),
        })

    # 1. Normalize column names.
    original_columns = list(cleaned.columns)
    normalized_columns = [
        str(column).strip()
        for column in original_columns
    ]

    if any(not column for column in normalized_columns):
        raise ValueError(
            "One or more column names are empty after trimming."
        )

    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError(
            "Cleaning would create duplicate column names. "
            "Resolve conflicting headers before continuing."
        )

    cleaned.columns = normalized_columns

    changed_headers = sum(
        old != new
        for old, new in zip(
            original_columns,
            normalized_columns,
        )
    )

    if changed_headers:
        record(
            "__columns__",
            "Trimmed whitespace from column names",
            changed_headers,
        )

    # 2. Replace positive and negative infinity with missing values.
    # Infinity can cause model fitting and scaling to fail.
    numeric_columns = cleaned.select_dtypes(
        include=["number"]
    ).columns

    for column in numeric_columns:
        series = cleaned[column]

        if not pd.api.types.is_numeric_dtype(series.dtype):
            continue

        try:
            infinite_mask = pd.Series(
                np.isinf(
                    series.to_numpy(
                        dtype=float,
                        na_value=np.nan,
                    )
                ),
                index=series.index,
            )
        except (TypeError, ValueError):
            continue

        infinite_count = int(infinite_mask.sum())

        if infinite_count:
            cleaned[column] = series.mask(
                infinite_mask,
                np.nan,
            )

            record(
                column,
                "Replaced infinite numeric values with missing values",
                infinite_count,
            )

    # 3. Clean text columns.
    for column in cleaned.columns:
        series = cleaned[column]

        if not _is_text_dtype(series):
            continue

        original = series.copy()
        normalized = _normalise_text(series)

        original_text = original.astype("string")

        whitespace_changes = int(
            (
                original_text.notna()
                & normalized.notna()
                & original_text.ne(normalized).fillna(False)
            ).sum()
        )

        if whitespace_changes:
            record(
                column,
                "Trimmed whitespace from text values",
                whitespace_changes,
            )

        original_missing = int(original.isna().sum())
        normalized_missing = int(normalized.isna().sum())

        marker_count = max(
            0,
            normalized_missing - original_missing,
        )

        if marker_count:
            record(
                column,
                "Converted recognized missing markers to missing values",
                marker_count,
            )

        cleaned[column] = normalized

    # 4. Convert numeric-looking text columns when conversion is safe.
    if normalize_numeric:
        for column in cleaned.columns:
            series = cleaned[column]

            if not _is_text_dtype(series):
                continue

            non_missing = series.notna()

            if not non_missing.any():
                continue

            # Do not convert identifier columns to numeric automatically.
            # IDs are identifiers, not meaningful continuous features.
            if str(column).strip().lower() in IDENTIFIER_NAMES:
                continue

            # Avoid converting common binary and categorical labels
            # into numbers merely because they contain numeric strings.
            candidate = series.loc[non_missing].astype("string")

            numeric_candidate = pd.to_numeric(
                candidate,
                errors="coerce",
            )

            successful = int(numeric_candidate.notna().sum())
            total = int(non_missing.sum())

            if total == 0:
                continue

            # Convert only when every non-missing value is numeric.
            if successful == total:
                converted = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                cleaned[column] = converted

                record(
                    column,
                    "Converted numeric-looking text to numeric values",
                    total,
                )

    # 5. Normalize categorical labels without changing their meaning.
    if normalize_categories:
        for column in cleaned.columns:
            series = cleaned[column]

            if not _is_text_dtype(series):
                continue

            # Only collapse repeated internal whitespace.
            # Do not lowercase values: labels can be case-sensitive.
            normalized = series.str.replace(
                r"\s+",
                " ",
                regex=True,
            )

            changed = int(
                (
                    series.notna()
                    & normalized.notna()
                    & series.ne(normalized).fillna(False)
                ).sum()
            )

            if changed:
                cleaned[column] = normalized

                record(
                    column,
                    "Normalized repeated internal whitespace in categories",
                    changed,
                )

    # 6. Report duplicate rows; preserve them unless the user decides
    # they are accidental duplicates.
    duplicate_count = int(cleaned.duplicated().sum())

    if duplicate_count:
        record(
            "__dataset__",
            "Detected duplicate rows; no rows were removed",
            duplicate_count,
        )

    # 7. Report constant columns that cannot help classification.
    for column in cleaned.columns:
        unique_count = int(cleaned[column].nunique(dropna=True))

        if unique_count <= 1:
            if unique_count == 0:
                action = "Column contains only missing values"
            else:
                action = (
                    "Constant column; consider excluding it from training"
                )

            record(
                column,
                action,
                len(cleaned),
            )

    # 8. Report potential numeric outliers using the IQR rule.
    # Do not remove or cap them automatically: valid medical values
    # can be extreme, and the target must not be altered blindly.
    if report_outliers:
        numeric_columns = cleaned.select_dtypes(
            include=["number"]
        ).columns

        for column in numeric_columns:
            values = cleaned[column].dropna()

            if len(values) < 4 or values.nunique() < 2:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            if not np.isfinite(iqr) or iqr <= 0:
                continue

            lower_bound = q1 - outlier_iqr_multiplier * iqr
            upper_bound = q3 + outlier_iqr_multiplier * iqr

            outlier_count = int(
                (
                    (values < lower_bound)
                    | (values > upper_bound)
                ).sum()
            )

            if outlier_count:
                record(
                    column,
                    (
                        "Potential IQR outliers detected "
                        f"(bounds: {lower_bound:.4g} to "
                        f"{upper_bound:.4g}); values retained"
                    ),
                    outlier_count,
                )

    # 9. Report missing values after safe cleaning.
    for column in cleaned.columns:
        missing_count = int(cleaned[column].isna().sum())

        if missing_count:
            record(
                column,
                "Missing values remain; imputation or review is needed",
                missing_count,
            )

    report_df = pd.DataFrame(
        report,
        columns=REPORT_COLUMNS,
    )

    return cleaned, report_df
