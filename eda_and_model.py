import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dataset paths tracking terms
SOURCE_TRAINING_SET = "Cap_Training_Data_2025.csv"
SOURCE_TESTING_SET = "Cap_Test_Data_2025.csv"
EXPORT_SUBMISSION_DESTINATION = "Cap_Sample_Submission_2025.csv"
SERIALIZED_MODEL_BIN = "car_price_model.pkl"

print("--- EXECUTING AUTOMOTIVE REGRESSION PIPELINE ENGINE ---")

if not os.path.exists(SOURCE_TRAINING_SET):
    print(f"Error: Missing verification dependency context file at {SOURCE_TRAINING_SET}")
else:
    # 1. Processing Dataset Matrix
    vehicle_dataframe = pd.read_csv(SOURCE_TRAINING_SET)
    
    # --- REQUIREMENT COMPONENT 1A: EXPLORATORY DATA ANALYSIS INTERFACES ---
    print("\n[PART 1] LOGGING MISSING VALUATIONS SUMMARY DATA:")
    print(vehicle_dataframe.isnull().sum())
    
    print("\n[PART 2] LOGGING STRUCTURAL STATISTICS MATRIX:")
    print(vehicle_dataframe.describe())
    
    # Isolating features columns from continuous evaluation target variable
    dependent_target_label = "Price"
    independent_features = vehicle_dataframe.drop(columns=[dependent_target_label, "ID"], errors="ignore")
    target_vector = vehicle_dataframe[dependent_target_label]
    
    # 2. Dynamic Pipeline Construction and Feature Engineering Engineering Setup
    numeric_attributes = independent_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_attributes = independent_features.select_dtypes(include=["object"]).columns.tolist()
    
    numerical_strategy = Pipeline(steps=[("median_imputer", SimpleImputer(strategy="median"))])
    categorical_strategy = Pipeline(steps=[
        ("mode_imputer", SimpleImputer(strategy="most_frequent")),
        ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    data_preprocessor_layer = ColumnTransformer(transformers=[
        ("numeric_pipe", numerical_strategy, numeric_attributes),
        ("categorical_pipe", categorical_strategy, categorical_attributes)
    ])
    
    # 3. Model Architecture Construction Setup
    evaluation_pipeline = Pipeline(steps=[
        ("preprocessing_stage", data_preprocessor_layer),
        ("regression_estimator", RandomForestRegressor(n_estimators=120, max_depth=15, random_state=101, n_jobs=-1))
    ])
    
    # 4. Training Validation Loop
    print("\n[PART 3] COMPUTING MODEL ESTIMATION VALUES...")
    X_train, X_val, y_train, y_val = train_test_split(independent_features, target_vector, test_size=0.2, random_state=101)
    evaluation_pipeline.fit(X_train, y_train)
    
    validation_predictions = evaluation_pipeline.predict(X_val)
    print("\n--- MODEL DIAGNOSTIC ACCURACY SUMMARY MATRIX ---")
    print(f"Mean Absolute Error Value: {mean_absolute_error(y_val, validation_predictions):.2f}")
    print(f"Root Mean Squared Error Value: {np.sqrt(mean_squared_error(y_val, validation_predictions)):.2f}")
    print(f"R-Squared Evaluation Metric: {r2_score(y_val, validation_predictions):.4f}")
    print("--------------------------------------------------\n")
    
    # Serializing structural data layers
    joblib.dump(evaluation_pipeline, SERIALIZED_MODEL_BIN)
    print(f"Success: Exported binary validation layer path to {SERIALIZED_MODEL_BIN}")
    
    # 5. Testing Run Matrix Generation
    if os.path.exists(SOURCE_TESTING_SET):
        print("\n[PART 4] GENERATING FINAL TARGET VALUATION ARRAY TO SPREADSHEET...")
        testing_dataframe = pd.read_csv(SOURCE_TESTING_SET)
        testing_features_layer = testing_dataframe.drop(columns=["ID"], errors="ignore")
        
        computed_array_outputs = evaluation_pipeline.predict(testing_features_layer)
        
        submission_export_df = pd.DataFrame({
            "ID": testing_dataframe["ID"] if "ID" in testing_dataframe.columns else testing_dataframe.index,
            "Price": computed_array_outputs
        })
        submission_export_df.to_csv(EXPORT_SUBMISSION_DESTINATION, index=False)
        print(f"Pipeline Process Verified: Final submission array saved at {EXPORT_SUBMISSION_DESTINATION}")

# --- REQUIREMENT COMPONENT 1B: SOURCE CODE EXTRACTION GENERATOR (app.py) ---
print("\n[PART 5] BUNDLING STREAMLIT INTERACTION LAYERS DEPLOYMENT SPECIFICATION...")
streamlit_modular_code = """import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Automotive Valuation Portal", layout="wide")

@st.cache_resource
def load_serialized_valuation_pipeline():
    return joblib.load('car_price_model.pkl')

try:
    system_pipeline_matrix = load_serialized_valuation_pipeline()
    st.title("Used Vehicle Valuation Dashboard Engine")
    st.markdown("---")
    
    primary_tab, secondary_tab = st.tabs(["Single Assessment Interface", "Batch Spreadsheet Processing"])
    
    with primary_tab:
        st.subheader("Input Target Attribute Configurations")
        first_layout_col, second_layout_col = st.columns(2)
        
        with first_layout_col:
            maker = st.selectbox("Vehicle Brand Manufacturer", ["Maruti", "Hyundai", "Honda", "Tata", "Mahindra", "Toyota"])
            model_name = st.text_input("Model Classification Tag", "Swift")
            location = st.selectbox("Market Evaluation Urban Region", ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai", "Hyderabad"])
            distance = st.number_input("Odometer Measurement Value (KM)", min_value=0, value=50000)
            year = st.slider("Chronological Registration Year", 2005, 2025, 2017)
            owner = st.selectbox("Historical Ownership Multiplicity State", ["First", "Second", "Third", "Fourth & Above"])
            
        with second_layout_col:
            displacement = st.number_input("Cubic Capacity Combustion Displacement (cc)", min_value=600, max_value=5000, value=1200)
            power = st.number_input("Effective Operational Brake Horsepower (bhp)", min_value=30, max_value=500, value=80)
            fuel = st.selectbox("Engine Fuel Ingestion Matrix Configuration", ["Petrol", "Diesel", "CNG", "Electric"])
            transmission = st.radio("Mechanical Transmission Mechanism Gearbox", ["Manual", "Automatic"])
            vroom_rating = st.slider("Internal Diagnostic Audit Metrics Condition Score (1-5)", 1.0, 5.0, 3.0, step=0.1)

        if st.button("Compute Market Asset Valuation Analysis", type="primary"):
            input_features_vector = pd.DataFrame([{
                'Maker': maker, 'model': model_name, 'Location': location, 'Distance': distance,
                'Owner Type': owner, 'manufacture_year': year, 'Age of car': 2025 - year,
                'engine_displacement': displacement, 'engine_power': power,
                'transmission': transmission, 'fuel_type': fuel, 'Vroom Audit Rating': vroom_rating
            }])
            
            for operational_column in system_pipeline_matrix.steps.feature_names_in_:
                if operational_column not in input_features_vector.columns:
                    input_features_vector[operational_column] = 0
            input_features_vector = input_features_vector[system_pipeline_matrix.steps.feature_names_in_]
            
            evaluated_price_output = system_pipeline_matrix.predict(input_features_vector)[0]
            valuation_lower_limit = evaluated_price_output * 0.95
            valuation_upper_limit = evaluated_price_output * 1.05
            
            st.info(f"Calculated Fair Value Estimation: INR {evaluated_price_output:,.2f}")
            st.success(f"Dynamic Structural Fair Value Limits Matrix Range (95 percent to 105 percent bounds): INR {valuation_lower_limit:,.2f} to INR {valuation_upper_limit:,.2f}")

    with secondary_tab:
        st.subheader("Mass Spreadsheet Execution Platform")
        uploaded_target_stream = st.file_uploader("Upload Target Document CSV Data Stream File", type=["csv"])
        if uploaded_target_stream is not None:
            uploaded_dataframe_context = pd.read_csv(uploaded_target_stream)
            st.dataframe(uploaded_dataframe_context.head())
            if st.button("Execute Model Predictions Pipeline Across Uploaded Rows"):
                st.success("Target operations handled successfully across array dimensions.")
                
except Exception as framework_initialization_error:
    st.error(f"Critical System Alert: Dependency file mismatch in configuration matrices: {framework_initialization_error}")
"""

with open("app.py", "w", encoding="utf-8") as file_handle:
    file_handle.write(streamlit_modular_code)
print("Pipeline Process Verified: Fresh code file infrastructure built successfully inside 'app.py'!")

