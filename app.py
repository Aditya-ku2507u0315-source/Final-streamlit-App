import streamlit as st
import pickle
import numpy as np

# --- 1. LOAD THE MODEL ---
# The @st.cache_resource decorator ensures the model is only loaded once, 
# making the app run much faster when users interact with it.
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# --- 2. BUILD THE UI ---
st.title("✈️ Travel Service Customer Churn Predictor")
st.write("Enter the customer's details below to predict if they are likely to churn (leave the service).")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    frequent_flyer = st.selectbox("Frequent Flyer?", ["No", "Yes"])
    services_opted = st.number_input("Number of Services Opted", min_value=0, max_value=10, value=1, step=1)

with col2:
    account_synced = st.selectbox("Account Synced to Social Media?", ["No", "Yes"])
    booked_hotel = st.selectbox("Booked Hotel?", ["No", "Yes"])
    # Present income as a single friendly dropdown, we will encode it behind the scenes
    income_class = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])

# --- 3. PROCESS INPUTS & PREDICT ---
if st.button("Predict Churn", type="primary"):
    
    # Map Yes/No to 1/0
    yes_no_map = {"Yes": 1, "No": 0}
    freq_flyer_val = yes_no_map[frequent_flyer]
    acct_synced_val = yes_no_map[account_synced]
    booked_hotel_val = yes_no_map[booked_hotel]
    
    # Handle One-Hot Encoding for Income
    # If High Income is selected, both Low and Middle become 0
    low_income_val = 1 if income_class == "Low Income" else 0
    middle_income_val = 1 if income_class == "Middle Income" else 0
    
    # Construct the final feature array in the EXACT order the model expects
    features = np.array([[
        age, 
        freq_flyer_val, 
        services_opted, 
        acct_synced_val, 
        booked_hotel_val, 
        low_income_val, 
        middle_income_val
    ]])
    
    # Make the prediction
    prediction = model.predict(features)
    
    # --- 4. DISPLAY RESULTS ---
    st.divider()
    if prediction[0] == 1:
        st.error("⚠️ **Prediction: Churn** - This customer is likely to leave the service.")
    else:
        st.success("✅ **Prediction: Stay** - This customer is likely to continue using the service.")