
"""Train and register disease-specific models."""

import streamlit as st

from src.model_training import train_disease_model, list_saved_models


st.set_page_config(
    page_title="Model Training | Medical AutoML",
    page_icon="🧠",
    layout="wide",
)

st.title("Disease Model Training")
st.write(
    "Train separate models for stroke, diabetes, heart disease, "
    "or other classification targets."
)

df = st.session_state.get("cleaned_dataset")
if df is None:
    df = st.session_state.get("loaded_dataset")

if df is None or df.empty:
    st.info("Upload a historical labeled dataset before training.")
    st.stop()

target_options = list(df.columns)
previous_target = st.session_state.get("selected_target")
target_index = (
    target_options.index(previous_target)
    if previous_target in target_options
    else 0
)

with st.form("train_disease_model_form"):
    disease_name = st.text_input(
        "Disease model name",
        placeholder="e.g. Stroke",
        help="Use a distinct name for each disease model.",
    )
    target_column = st.selectbox(
        "Target label column",
        target_options,
        index=target_index,
    )
    test_size = st.slider(
        "Held-out test proportion",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
    )
    submitted = st.form_submit_button(
        "Train and save disease model",
        type="primary",
    )

if submitted:
    try:
        with st.spinner("Training models and preparing explanations..."):
            output = train_disease_model(
                df=df,
                target_column=target_column,
                disease_name=disease_name,
                test_size=test_size,
            )

        package = output["package"]
        st.session_state["last_trained_model_slug"] = package["model_slug"]

        st.success(
            f"Saved {package['disease_name']} model to "
            f"{output['model_path']}"
        )

        st.subheader("Model comparison")
        st.dataframe(
            output["results"].round(4),
            width="stretch",
            hide_index=True,
        )

        metrics = package["metrics"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric(
            "Balanced accuracy",
            f"{metrics['balanced_accuracy']:.3f}",
        )
        col3.metric("Macro F1", f"{metrics['macro_f1']:.3f}")

        if output["errors"]:
            with st.expander("Models that could not be trained"):
                for name, error in output["errors"].items():
                    st.write(f"**{name}:** {error}")

    except Exception as exc:
        st.error(f"Training failed: {exc}")

st.divider()
st.subheader("Saved disease models")

saved_models = list_saved_models()

if not saved_models:
    st.info("No disease models have been saved yet.")
else:
    st.dataframe(
        [
            {
                "Disease": item["name"],
                "Target column": item["target"],
                "Selected algorithm": item["model"],
                "Artifact": item["path"],
            }
            for item in saved_models
        ],
        width="stretch",
        hide_index=True,
    )
