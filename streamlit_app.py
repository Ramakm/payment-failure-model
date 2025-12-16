import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime

# -----------------------
# Setup & Helper Functions
# -----------------------

# Set page config
st.set_page_config(
    page_title="Payment Failure Prediction",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("payment_failure_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run `train.py` first to generate `payment_failure_model.pkl`.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_input(df):
    """
    Applies the same feature engineering as train.py.
    """
    # Defensive copy
    df = df.copy()
    
    current_year = datetime.now().year
    
    # Calculate age
    if 'dateOfBirth' in df.columns:
        df["age"] = current_year - df["dateOfBirth"].astype(str).str[:4].astype(int)
    
    # Map ID verification
    if 'idVerificationStatus' in df.columns:
        df["id_verified"] = df["idVerificationStatus"].map({"Y": 1, "N": 0})
        # Handle cases where mapping might produce NaN if unexpected values
        df["id_verified"] = df["id_verified"].fillna(0).astype(int)

    # Calculate cross_border
    # Ensure receiver.address.countryCode exists. In batch JSON it might be nested dicts, 
    # but based on train.py it expects "receiver.address.countryCode" column after json_normalize
    # Or if we pass a dict, we need to flatten it or extract it.
    
    # For the UI input, we ask for 'receiver_address_countryCode' directly.
    # For batch input (JSON), we might need to normalize.
    
    # Let's handle both "receiver.address.countryCode" (from json_normalize) 
    # and a manual column if constructed differently.
    
    col_receiver = "receiver.address.countryCode"
    if col_receiver not in df.columns:
        # Check if we have 'receiver' column that is a dict (batch scenario before normalize?)
        pass 
        # For simplicity, assume the input DF to this function has flattening applied if coming from JSON
        # OR has the specific column names we set in the UI.

    if col_receiver in df.columns and 'countryOfBirth' in df.columns:
         df["cross_border"] = (
            df["countryOfBirth"] != df[col_receiver]
        ).astype(int)
    
    # Select columns expected by the pipeline
    # The pipeline expects: ["occupation", "purposeOfTransaction", "sourceOfFunds", 
    # "countryOfBirth", "nationality", "receiver.address.countryCode", "age", "id_verified", "cross_border"]
    
    expected_cols = [
        "occupation",
        "purposeOfTransaction",
        "sourceOfFunds",
        "countryOfBirth",
        "nationality",
        "receiver.address.countryCode",
        "age",
        "id_verified",
        "cross_border"
    ]
    
    # Add missing columns with defaults if necessary (not doing it here to be strict)
    
    # Filter to only expected columns
    return df[expected_cols]

# -----------------------
# Streamlit App
# -----------------------

def main():
    st.title("💳 Payment Failure Prediction")
    st.markdown("""
    Predict whether a payment transaction will fail based on customer KYC information and transaction metadata.
    """)
    
    model = load_model()
    
    if not model:
        st.warning("Model not loaded. App functionality will be limited.")
        st.stop()

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    # -----------------------
    # Tab 1: Single Prediction
    # -----------------------
    with tab1:
        st.header("Single Transaction Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            occupation = st.selectbox("Occupation", ["worker", "employee", "engineer", "teacher", "civil_servant", "other"])
            purpose = st.selectbox("Purpose of Transaction", ["bills", "education", "shopping", "medical", "investment", "other"])
            source = st.selectbox("Source of Funds", ["Cash", "Bank Transfer", "Credit Card", "Other"])
            dob = st.date_input("Date of Birth", value=datetime(1990, 1, 1))
        
        with col2:
            country = st.text_input("Country of Birth", value="IN", max_chars=2)
            nationality = st.text_input("Nationality", value="IN", max_chars=2)
            receiver_country = st.text_input("Receiver Country Code", value="US", max_chars=2)
            id_verified = st.radio("ID Verified?", ["Yes", "No"], index=0)
        
        if st.button("Predict", type="primary"):
            # Construct DataFrame
            input_data = {
                "occupation": [occupation],
                "purposeOfTransaction": [purpose],
                "sourceOfFunds": [source],
                "countryOfBirth": [country],
                "nationality": [nationality],
                "receiver.address.countryCode": [receiver_country],
                "dateOfBirth": [str(dob)], # preprocess expects string YYYY-...
                "idVerificationStatus": ["Y" if id_verified == "Yes" else "N"]
            }
            
            raw_df = pd.DataFrame(input_data)
            processed_df = preprocess_input(raw_df)
            
            try:
                prediction = model.predict(processed_df)[0]
                proba = model.predict_proba(processed_df)[0][1]
                
                if prediction == 1:
                    st.error(f"⚠️ Payment Failure Risk Detected! (Probability: {proba:.2%})")
                else:
                    st.success(f"✅ Payment Likely Successful. (Failure Probability: {proba:.2%})")
                    
                with st.expander("Show Processed Data"):
                    st.dataframe(processed_df)
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # -----------------------
    # Tab 2: Batch Prediction
    # -----------------------
    with tab2:
        st.header("Batch Prediction")
        
        uploaded_file = st.file_uploader("Upload a JSON file (containing a list of records)", type=["json"])
        
        if uploaded_file is not None:
            try:
                # Read JSON
                batch_data = json.load(uploaded_file)
                
                # Check if list or dict
                if isinstance(batch_data, dict):
                    batch_data = [batch_data]
                    
                # Normalize (flattens nested 'receiver.address.countryCode')
                raw_df = pd.json_normalize(batch_data)
                
                st.write(f"Loaded {len(raw_df)} records.")
                
                if st.button("Run Batch Prediction"):
                    processed_df = preprocess_input(raw_df)
                    
                    predictions = model.predict(processed_df)
                    probas = model.predict_proba(processed_df)[:, 1]
                    
                    # Attach results to original dataframe for display
                    # Using raw_df to show original context + results
                    results_df = raw_df.copy()
                    results_df["Predicted_Failure"] = predictions
                    results_df["Failure_Probability"] = probas
                    
                    st.dataframe(results_df)
                    
                    # Download CSV
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='prediction_results.csv',
                        mime='text/csv',
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
