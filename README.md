# Indian Pre-Owned Car Valuation Engine

An end-to-end Machine Learning pipeline utilizing an optimized Random Forest Regressor framework to estimate used vehicle market values based on historical attributes. The deployment layer features an interactive dashboard built natively using Streamlit.

## Project Deliverables
- **Component 1A (Model Pipeline & EDA)**: Core data cleaning script (`eda_and_model.py`) that profiles data, handles missing attributes via robust median/mode imputation, maps categorical features with One-Hot Encoding, and runs model training logic.
- **Component 1B (Deployment Application)**: Graphical interface script (`app.py`) providing single vehicle asset valuations applying a dynamic ±5% fair valuation range band and a bulk inference page.
- **Submission Output**: Formatted evaluation dataset containing target prices saved under `Cap_Sample_Submission_2025.csv`.

## Preprocessing & Pipeline Flow
1. **Missing Data Imputation**: Auto-replaces numerical null elements with descriptive column medians and categorical gaps with column modes.
2. **Feature Optimization**: Transforms high-cardinality nominal text strings (e.g., Maker, Location, Fuel Type) into clean numeric tracking structures via scikit-learn One-Hot Encoders.
3. **Model Selection**: Harnesses an ensemble Random Forest Regressor to safely avoid overfitting while identifying subtle error boundaries in cross-feature parameters.

## Local Installation & Environment Setup

Ensure you have your virtual environment activated, then run the terminal installation script below:
```bash
pip install pandas numpy scikit-learn joblib streamlit
```

## Running the Architecture

To run the full Exploratory Data Analysis (EDA) dashboard routine, execute data processing rules, train the ML model, and generate your spreadsheet outputs:
```bash
python eda_and_model.py
```

To boot up the interactive client-facing Streamlit application engine panel inside your local web browser (`http://localhost:8501`):
```bash
streamlit run app.py
```

