
"""Medical AutoML application home page."""

import streamlit as st


st.set_page_config(
    page_title="Medical AutoML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Medical AutoML")
st.subheader("Medical dataset analysis and machine learning")

st.write(
    "A workflow for uploading medical datasets, cleaning data, "
    "training classification models, evaluating performance, "
    "and exploring model explanations."
)

st.warning(
    "Research and development use only. This application is not a "
    "diagnostic device and must not be used to make clinical decisions "
    "without appropriate validation, governance, and professional oversight."
)

st.divider()

st.header("Application workflow")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Data preparation")
    st.markdown(
        """
        **Dataset Upload**
        - Upload CSV or Excel datasets
        - Review columns, missing values, and duplicate rows
        - Select the target label

        **Data Cleaning**
        - Normalize column names and whitespace
        - Identify common missing-value markers
        - Review the cleaning report
        """
    )

    st.subheader("2. Model development")
    st.markdown(
        """
        **Model Training**
        - Compare classification algorithms
        - Review evaluation metrics
        - Save the selected model pipeline
        """
    )

with col2:
    st.subheader("3. Model usage")
    st.markdown(
        """
        **Prediction**
        - Upload new records
        - Generate model outputs
        - Export results to CSV

        **Explainability**
        - Review global feature importance
        - Explore individual SHAP explanations
        """
    )

    st.subheader("Current project status")

    status_items = [
        ("Dataset ingestion", "pages/1_Dataset_Upload.py"),
        ("Data cleaning", "pages/2_Data_Cleaning.py"),
        ("Model training", "pages/3_Model_Training.py"),
        ("Prediction", "pages/4_Prediction.py"),
        ("Explainability", "pages/5_Explainability.py"),
    ]

    for label, page_path in status_items:
        st.write(f"**{label}:** `{page_path}`")

st.divider()

st.header("Getting started")

st.markdown(
    """
    1. Open **Dataset Upload** from the sidebar.
    2. Upload your dataset and choose the target column.
    3. Open **Data Cleaning** and review the cleaning report.
    4. Open **Model Training** to train and compare classifiers.
    5. Use **Prediction** or **Explainability** after training.
    """
)

st.info(
    "Only use data you are authorized to process. Remove direct "
    "patient identifiers where possible, protect sensitive health "
    "information, and use trusted model artifacts."
)

st.caption("Medical AutoML | Experimental machine learning platform")
