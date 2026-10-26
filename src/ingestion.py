
"""Dataset ingestion and profiling for Medical AutoML."""

from io import BytesIO
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

MISSING_TEXT_VALUES = {"", "na", "n/a", "null", "none", "nan"}


def get_file_extension(filename: str) -> str:
    """Return a normalized file extension."""
    return Path(filename).suffix.lower()


def validate_file_extension(filename: str) -> None:
    """Reject unsupported file types."""
    extension = get_file_extension(filename)

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type: {extension or '(no extension)'}. "
            f"Supported extensions: {supported}"
        )


def validate_column_names(columns) -> None:
    """Reject blank or duplicate column names."""
    names = [str(column).strip() for column in columns]

    blank_columns = [
        index for index, name in enumerate(names) if not name
    ]

    if blank_columns:
        raise ValueError(
            f"Blank column names detected at positions: {blank_columns}"
        )

    duplicates = sorted(
        {name for name in names if names.count(name) > 1}
    )

    if duplicates:
        raise ValueError(
            f"Duplicate column names were detected: {duplicates}"
        )


def get_excel_sheet_names(file_bytes: bytes) -> list[str]:
    """Return sheet names from an Excel workbook."""
    try:
        with pd.ExcelFile(BytesIO(file_bytes)) as workbook:
            return workbook.sheet_names
    except Exception as exc:
        raise ValueError(
            f"Could not read the Excel workbook: {exc}"
        ) from exc


def load_dataset(
    file_bytes: bytes,
    filename: str,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    """Load a CSV or Excel dataset from bytes."""
    validate_file_extension(filename)

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    extension = get_file_extension(filename)

    try:
        if extension == ".csv":
            df = pd.read_csv(
                BytesIO(file_bytes),
                sep=None,
                engine="python",
                encoding="utf-8-sig",
            )

        elif extension in {".xlsx", ".xls"}:
            if sheet_name is None:
                df = pd.read_excel(BytesIO(file_bytes))
            else:
                df = pd.read_excel(
                    BytesIO(file_bytes),
                    sheet_name=sheet_name,
                )

    except Exception as exc:
        raise ValueError(
            f"Could not load '{filename}': {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "The dataset has no data rows or no readable columns."
        )

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    validate_column_names(df.columns)

    return df


def _missing_mask(series: pd.Series) -> pd.Series:
    """Identify actual missing values and common text placeholders."""
    mask = series.isna()

    if (
        pd.api.types.is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.StringDtype)
    ):
        text = series.astype("string").str.strip().str.lower()
        mask = mask | text.isin(MISSING_TEXT_VALUES)

    return mask


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate dataset statistics and a column-level profile."""
    if df.empty or len(df.columns) == 0:
        raise ValueError("Cannot profile an empty dataset.")

    column_profiles = []

    for column in df.columns:
        series = df[column]
        missing_mask = _missing_mask(series)
        missing_count = int(missing_mask.sum())

        observed = series.loc[~missing_mask]
        unique_count = int(observed.nunique(dropna=True))

        sample_values = (
            observed.drop_duplicates()
            .head(5)
            .map(lambda value: str(value)[:100])
            .tolist()
        )

        column_profiles.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "non_missing": int(len(df) - missing_count),
                "missing_count": missing_count,
                "missing_pct": round(
                    missing_count / len(df) * 100, 2
                ),
                "unique_count": unique_count,
                "sample_values": ", ".join(sample_values),
            }
        )

    profiles_df = pd.DataFrame(column_profiles)

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(
            profiles_df["missing_count"].sum()
        ),
        "columns_with_missing": int(
            (profiles_df["missing_count"] > 0).sum()
        ),
        "column_profiles": profiles_df,
    }
