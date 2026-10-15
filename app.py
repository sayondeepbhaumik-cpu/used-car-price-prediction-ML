import streamlit as st
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
            distance_val = st.number_input("Odometer Measurement Value (KM)", min_value=0, value=50000)
            year = st.slider("Chronological Registration Year", 2005, 2025, 2017)
            owner = st.selectbox("Historical Ownership Multiplicity State", ["First", "Second", "Third", "Fourth & Above"])
            
        with second_layout_col:
            displacement = st.number_input("Cubic Capacity Combustion Displacement (cc)", min_value=600, max_value=5000, value=1200)
            power = st.number_input("Effective Operational Brake Horsepower (bhp)", min_value=30, max_value=500, value=80)
            fuel = st.selectbox("Engine Fuel Ingestion Matrix Configuration", ["Petrol", "Diesel", "CNG", "Electric"])
            transmission = st.radio("Mechanical Transmission Mechanism Gearbox", ["Manual", "Automatic"])
            vroom_rating = st.slider("Internal Diagnostic Audit Metrics Condition Score (1-5)", 1.0, 5.0, 3.0, step=0.1)
            
            body_type = st.selectbox("Vehicle Body Type Style", ["Hatchback", "Sedan", "SUV", "MUV", "Coupe"])
            seat_count = st.slider("Total Passenger Seat Count Layout", 2, 8, 5)
            door_count = st.slider("Total Vehicle Side Door Entry Count", 2, 5, 4)

        if st.button("Compute Market Asset Valuation Analysis", type="primary"):
            input_features_vector = pd.DataFrame([{
                'Maker': maker, 
                'model': model_name, 
                'Location': location, 
                'Distance ': distance_val,  
                'Owner Type': owner, 
                'manufacture_year': year, 
                'Age of car': 2025 - year,
                'engine_displacement': displacement, 
                'engine_power': power,
                'transmission': transmission, 
                'fuel_type': fuel, 
                'Vroom Audit Rating': vroom_rating,
                'body_type': body_type,
                'seat_count': seat_count,
                'door_count': door_count
            }])
            
            try:
                evaluated_price_output = system_pipeline_matrix.predict(input_features_vector)[0]
                valuation_lower_limit = evaluated_price_output * 0.95
                valuation_upper_limit = evaluated_price_output * 1.05
                
                st.info(f"Calculated Fair Value Estimation: INR {evaluated_price_output:,.2f}")
                st.success(f"Dynamic Structural Fair Value Limits Matrix Range (95 percent to 105 percent bounds): INR {valuation_lower_limit:,.2f} to INR {valuation_upper_limit:,.2f}")
            except Exception as prediction_error:
                st.error(f"Error mapping attributes into pipeline fields: {prediction_error}")

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

