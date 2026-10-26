# MediShield 🛡️

### Medical Machine Learning and Prediction Platform

MediShield is a Python-based medical machine learning application designed to train disease prediction models, generate predictions from patient CSV files, evaluate model performance, and explore model explainability through an interactive Streamlit interface.

> **Disclaimer:** MediShield is an educational and research project. It is not a medical device and must not be used to diagnose diseases, make treatment decisions, or replace advice from qualified healthcare professionals.

## Features

- **Dataset Management:** Load and prepare datasets for machine learning.
- **Data Cleaning:** Handle missing values and prepare data for modeling.
- **Model Training:** Train and compare machine learning classifiers.
- **Disease Prediction:** Generate predictions from uploaded CSV files.
- **Model Evaluation:** Review classification metrics and model performance.
- **Explainable AI:** Explore feature importance and individual predictions using SHAP.
- **Interactive Dashboard:** Use a Streamlit interface to access the application's functionality.
- **Multiple Disease Models:** Designed to support separate models for stroke, diabetes, and heart disease.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- SHAP
- Plotly
- Matplotlib and Seaborn
- Joblib

## Project Structure

```text
MediShield/
├── app.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   └── models/
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── prediction.py
│   ├── explainability.py
│   └── visualizations.py
├── pages/
│   ├── 1_Dataset_Upload.py
│   ├── 2_Data_Cleaning.py
│   ├── 3_Model_Training.py
│   ├── 4_Prediction.py
│   └── 5_Explainability.py
├── tests/
├── .gitignore
└── README.md
```

*The structure above describes the intended project layout; individual files and features may vary as development continues.*

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/MediShield.git
cd MediShield
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL to open the application in your browser.

## Using MediShield

1. Prepare a labeled dataset for the disease you want to model.
2. Load and clean the dataset.
3. Select the target column and train a disease-specific model.
4. Upload a CSV containing the required input features for prediction.
5. Review the generated predictions and available evaluation metrics.
6. Explore explanations for the predictions when supported by the trained model.

Column names and data types in prediction files must match the feature schema expected by the selected model.

## Datasets

For development and experimentation, publicly available datasets may be obtained from:

- [Stroke Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- [Pima Indians Diabetes Database — Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Heart Disease UCI — Kaggle](https://www.kaggle.com/datasets/uciml/heart-disease-uci)

Check each dataset's license and terms of use before redistribution. Public datasets may contain biases and may not represent real-world clinical populations.

## Data Privacy and Security

- Do not commit identifiable patient records or confidential medical data.
- Do not upload credentials, API keys, or secrets to GitHub.
- Keep large datasets and generated model artifacts out of version control unless there is a specific, safe reason to include them.
- Use appropriately anonymized or synthetic data for demonstrations.

## Current Development Status

MediShield is under active development. Training performance, model compatibility, data validation, and prediction explainability are being improved.

Model performance depends on dataset quality, preprocessing, class balance, and evaluation methodology. Reported metrics do not establish clinical validity.

## Future Improvements

- Faster training for large datasets.
- More robust input validation and error handling.
- Improved model comparison and evaluation reports.
- Consistent prediction and explainability workflows.
- Reproducible training and model versioning.
- Deployment and usability improvements.

## License

No license has been specified yet. Until a license is added, assume that the repository's contents are subject to the default applicable copyright rules. Add a suitable open-source license if you intend to permit reuse or redistribution.

---

**MediShield — Machine learning for medical data exploration.**
