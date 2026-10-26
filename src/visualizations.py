
"""Interactive visualizations for Medical AutoML."""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


def plot_model_comparison(results: pd.DataFrame) -> go.Figure:
    """Compare model accuracy and macro F1 scores."""
    required = {"model", "accuracy", "macro_f1"}

    if results.empty or not required.issubset(results.columns):
        raise ValueError(
            "Results must contain model, accuracy, and macro_f1 columns."
        )

    chart_data = results.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_f1"],
        var_name="metric",
        value_name="score",
    )

    fig = px.bar(
        chart_data,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Model Performance Comparison",
        labels={
            "model": "Model",
            "score": "Score",
            "metric": "Metric",
        },
        hover_data={"score": ":.4f"},
    )

    fig.update_layout(
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis_title="Classification Model",
        legend_title="Metric",
        template="plotly_white",
    )

    return fig


def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
) -> go.Figure:
    """Display a confusion matrix as an interactive heatmap."""
    if confusion_df.empty:
        raise ValueError("The confusion matrix is empty.")

    fig = px.imshow(
        confusion_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels={
            "x": "Predicted Class",
            "y": "Actual Class",
            "color": "Count",
        },
    )

    fig.update_layout(template="plotly_white")

    return fig


def plot_class_distribution(
    target: pd.Series,
) -> go.Figure:
    """Visualize the number of observations in each target class."""
    if target.empty:
        raise ValueError("The target series is empty.")

    counts = (
        target.astype("string")
        .fillna("(missing)")
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )

    fig = px.bar(
        counts,
        x="class",
        y="count",
        title="Target Class Distribution",
        labels={"class": "Class", "count": "Number of Rows"},
        text="count",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Target Class",
        yaxis_title="Number of Rows",
    )

    return fig


def plot_feature_importance(
    feature_names: list[str],
    importance_values,
    top_n: int = 20,
) -> go.Figure:
    """Display feature importance values when supported by a model."""
    values = list(importance_values)

    if len(feature_names) != len(values):
        raise ValueError(
            "Feature names and importance values must have equal lengths."
        )

    if not values:
        raise ValueError("No feature importance values were provided.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
        }
    )

    importance_df = (
        importance_df.sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {min(top_n, len(importance_df))} Important Features",
        labels={
            "feature": "Feature",
            "importance": "Importance",
        },
    )

    fig.update_layout(
        template="plotly_white",
        yaxis_title="",
        xaxis_title="Importance",
    )

    return fig
