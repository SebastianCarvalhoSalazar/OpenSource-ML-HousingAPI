"""
Streamlit web interface for Boston Housing price prediction.
Updated for Boston Housing Dataset (13 features).
"""
import streamlit as st
import requests
import pandas as pd
# import numpy as np
import plotly.express as px
# import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Boston Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# API URL
# API_URL = "http://localhost:8000"  # Change to "http://api:8000" for Docker
API_URL = "http://api:8000" # for Docker

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.prediction-result {
    background-color: #d4edda;
    border: 2px solid #c3e6cb;
    color: #155724;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    font-size: 1.2rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def predict_price(features):
    """Call API to predict house price."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


def main():
    # Header
    st.markdown('<p class="main-header">🏠 Boston Housing Price Predictor</p>', unsafe_allow_html=True)
    st.markdown("### Predict house prices using machine learning")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Information")
        
        # Check API status
        health = check_api_health()
        if health:
            st.success("✅ API Connected")
            st.info(f"**Model:** {health['model_version']}")
            
            if health.get('model_metrics'):
                st.markdown("**Model Performance:**")
                metrics = health['model_metrics']
                st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
                st.metric("RMSE", f"${metrics.get('rmse', 0):.2f}k")
                st.metric("MAE", f"${metrics.get('mae', 0):.2f}k")
        else:
            st.error("❌ API Not Available")
            st.warning("Make sure the API is running:\n```\npython app/api.py\n```")
        
        st.markdown("---")
        
        st.markdown("""
        ### 📖 How to use:
        1. Enter house features
        2. Click 'Predict Price'
        3. View prediction results
        
        ### 🔍 Features:
        - Real-time predictions
        - Interactive visualizations
        - Batch predictions
        - Model performance metrics
        """)
        
        # Quick examples
        st.markdown("### 🎯 Example Scenarios:")
        if st.button("Luxury Home"):
            st.session_state.example = "luxury"
        if st.button("Average Home"):
            st.session_state.example = "average"
        if st.button("Budget Home"):
            st.session_state.example = "budget"
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🏡 House Features (Boston Dataset)")
        
        # Set default values based on example selection
        if 'example' in st.session_state:
            if st.session_state.example == "luxury":
                defaults = {
                    'CRIM': 0.01, 'ZN': 80.0, 'INDUS': 1.0, 'CHAS': 1,
                    'NOX': 0.40, 'RM': 8.0, 'AGE': 10.0, 'DIS': 8.0,
                    'RAD': 1, 'TAX': 200.0, 'PTRATIO': 12.0, 'B': 396.9, 'LSTAT': 2.0
                }
            elif st.session_state.example == "budget":
                defaults = {
                    'CRIM': 5.0, 'ZN': 0.0, 'INDUS': 18.0, 'CHAS': 0,
                    'NOX': 0.65, 'RM': 5.0, 'AGE': 90.0, 'DIS': 2.0,
                    'RAD': 24, 'TAX': 666.0, 'PTRATIO': 20.0, 'B': 350.0, 'LSTAT': 20.0
                }
            else:  # average
                defaults = {
                    'CRIM': 0.2, 'ZN': 18.0, 'INDUS': 10.0, 'CHAS': 0,
                    'NOX': 0.55, 'RM': 6.5, 'AGE': 50.0, 'DIS': 4.0,
                    'RAD': 5, 'TAX': 400.0, 'PTRATIO': 16.0, 'B': 390.0, 'LSTAT': 10.0
                }
        else:
            defaults = {
                'CRIM': 0.00632, 'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
                'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
                'RAD': 1, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90, 'LSTAT': 4.98
            }
        
        # Input form
        with st.form("prediction_form"):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                crim = st.number_input(
                    "CRIM - Crime Rate",
                    min_value=0.0,
                    max_value=100.0,
                    value=defaults['CRIM'],
                    step=0.01,
                    help="Per capita crime rate by town",
                    format="%.5f"
                )
                
                zn = st.number_input(
                    "ZN - Residential Zoning",
                    min_value=0.0,
                    max_value=100.0,
                    value=defaults['ZN'],
                    step=1.0,
                    help="Proportion of residential land zoned for large lots"
                )
                
                indus = st.number_input(
                    "INDUS - Non-Retail Business",
                    min_value=0.0,
                    max_value=30.0,
                    value=defaults['INDUS'],
                    step=0.1,
                    help="Proportion of non-retail business acres"
                )
                
                chas = st.selectbox(
                    "CHAS - Charles River",
                    options=[0, 1],
                    index=defaults['CHAS'],
                    help="1 if tract bounds river, 0 otherwise"
                )
                
                nox = st.number_input(
                    "NOX - Nitric Oxides",
                    min_value=0.0,
                    max_value=1.0,
                    value=defaults['NOX'],
                    step=0.001,
                    help="Nitric oxides concentration (parts per 10M)",
                    format="%.3f"
                )
            
            with col_b:
                rm = st.number_input(
                    "RM - Avg Rooms",
                    min_value=1.0,
                    max_value=15.0,
                    value=defaults['RM'],
                    step=0.1,
                    help="Average number of rooms per dwelling"
                )
                
                age = st.number_input(
                    "AGE - Property Age",
                    min_value=0.0,
                    max_value=100.0,
                    value=defaults['AGE'],
                    step=1.0,
                    help="Proportion of units built before 1940"
                )
                
                dis = st.number_input(
                    "DIS - Employment Distance",
                    min_value=0.0,
                    max_value=15.0,
                    value=defaults['DIS'],
                    step=0.1,
                    help="Weighted distances to employment centers"
                )
                
                rad = st.number_input(
                    "RAD - Highway Access",
                    min_value=1,
                    max_value=24,
                    value=defaults['RAD'],
                    step=1,
                    help="Index of accessibility to radial highways"
                )
            
            with col_c:
                tax = st.number_input(
                    "TAX - Property Tax",
                    min_value=0.0,
                    max_value=800.0,
                    value=defaults['TAX'],
                    step=1.0,
                    help="Full-value property tax rate per $10,000"
                )
                
                ptratio = st.number_input(
                    "PTRATIO - Pupil-Teacher Ratio",
                    min_value=10.0,
                    max_value=25.0,
                    value=defaults['PTRATIO'],
                    step=0.1,
                    help="Pupil-teacher ratio by town"
                )
                
                b = st.number_input(
                    "B - Black Proportion",
                    min_value=0.0,
                    max_value=400.0,
                    value=defaults['B'],
                    step=1.0,
                    help="1000(Bk - 0.63)^2 where Bk is proportion of Black residents"
                )
                
                lstat = st.number_input(
                    "LSTAT - Lower Status %",
                    min_value=0.0,
                    max_value=40.0,
                    value=defaults['LSTAT'],
                    step=0.1,
                    help="Percentage of lower status population"
                )
            
            submitted = st.form_submit_button("🔮 Predict Price", type="primary")
            
            if submitted:
                # Prepare features for Boston Housing Dataset
                features = {
                    'CRIM': float(crim),
                    'ZN': float(zn),
                    'INDUS': float(indus),
                    'CHAS': int(chas),
                    'NOX': float(nox),
                    'RM': float(rm),
                    'AGE': float(age),
                    'DIS': float(dis),
                    'RAD': int(rad),
                    'TAX': float(tax),
                    'PTRATIO': float(ptratio),
                    'B': float(b),
                    'LSTAT': float(lstat)
                }
                
                # Make prediction
                with st.spinner("Predicting..."):
                    time.sleep(0.8)  # Visual feedback
                    result = predict_price(features)
                    time.sleep(0.2)  # Visual feedback
                    
                    if result:
                        st.session_state.last_prediction = result
                        st.session_state.last_features = features
                        st.balloons()
                    else:
                        st.error("Failed to get prediction. Check API logs.")
    
    with col2:
        st.header("📈 Prediction Result")
        
        if 'last_prediction' in st.session_state:
            result = st.session_state.last_prediction
            
            # Display prediction
            predicted_price = result['predicted_price']
            predicted_dollars = result['predicted_price_dollars']
            
            st.markdown(f"""
            <div class="prediction-result">
                <h2>💰 Predicted Price</h2>
                <h1>${predicted_dollars:,.0f}</h1>
                <p>(${predicted_price:.2f}k)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional info
            st.info(f"**Model:** {result['model_version']}")
            st.caption(f"Prediction time: {result['prediction_time']}")
        else:
            st.info("👈 Enter house features and click 'Predict Price' to see results")
    
    # Batch prediction section
    with st.expander("🔢 Batch Predictions", expanded=False):
        st.markdown("### Upload CSV or Excel for Batch Predictions")
        st.markdown("**Required columns:** CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT")
        
        # Sample data
        sample_data = pd.DataFrame([
            {
                'CRIM': 0.00632, 'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
                'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
                'RAD': 1, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.90, 'LSTAT': 4.98
            },
            {
                'CRIM': 0.02731, 'ZN': 0.0, 'INDUS': 7.07, 'CHAS': 0,
                'NOX': 0.469, 'RM': 6.421, 'AGE': 78.9, 'DIS': 4.9671,
                'RAD': 2, 'TAX': 242.0, 'PTRATIO': 17.8, 'B': 396.90, 'LSTAT': 9.14
            }
        ])
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            # Download sample CSV
            st.download_button(
                label="📥 Download Sample CSV",
                data=sample_data.to_csv(index=False),
                file_name="boston_housing_sample.csv",
                mime="text/csv"
            )
        
        with col_download2:
            # Download sample Excel
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                sample_data.to_excel(writer, index=False, sheet_name='Boston Housing')
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Sample Excel",
                data=buffer,
                file_name="boston_housing_sample.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # Read file based on extension
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                else:
                    st.error("Unsupported file format")
                    return
                
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("Predict All"):
                    with st.spinner(f"Processing {len(df)} predictions..."):
                        houses = df.to_dict('records')
                        
                        try:
                            response = requests.post(
                                f"{API_URL}/predict/batch",
                                json={"houses": houses},
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                results = response.json()
                                
                                # Add predictions to dataframe
                                df['Predicted_Price_k'] = [r['predicted_price'] for r in results]
                                df['Predicted_Price_Dollars'] = [r['predicted_price_dollars'] for r in results]
                                
                                st.success(f"✅ Predicted prices for {len(df)} houses")
                                st.dataframe(df)
                                
                                # Download button for results
                                result_buffer = io.BytesIO()
                                with pd.ExcelWriter(result_buffer, engine='openpyxl') as writer:
                                    df.to_excel(writer, index=False, sheet_name='Predictions')
                                result_buffer.seek(0)
                                
                                col_dl1, col_dl2 = st.columns(2)
                                
                                with col_dl1:
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download CSV",
                                        data=csv,
                                        file_name=f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv",
                                        mime="text/csv"
                                    )
                                
                                with col_dl2:
                                    st.download_button(
                                        label="📥 Download Excel",
                                        data=result_buffer,
                                        file_name=f"predictions_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                
                                # Summary statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Average Price", f"${df['Predicted_Price_Dollars'].mean():,.0f}")
                                with col2:
                                    st.metric("Min Price", f"${df['Predicted_Price_Dollars'].min():,.0f}")
                                with col3:
                                    st.metric("Max Price", f"${df['Predicted_Price_Dollars'].max():,.0f}")
                                
                                # Distribution plot
                                fig = px.histogram(
                                    df,
                                    x='Predicted_Price_Dollars',
                                    nbins=30,
                                    title='Distribution of Predicted Prices'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(f"API Error: {response.status_code} - {response.text}")
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
                
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    # Model information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Boston Housing Price Predictor v1.0</strong></p>
        <p>Built with Streamlit, FastAPI, and scikit-learn</p>
        <p>MLOps Project - Reproducible ML Pipeline</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()