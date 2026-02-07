import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("breast_cancer_model.pkl")

st.title("Breast Cancer Prediction App")

# Collect inputs on main page
input_data = []
for i in range(30):
    value = st.number_input(f"Feature {i+1}")
    input_data.append(value)

# Sidebar section
st.sidebar.header("Prediction Result")

if st.sidebar.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)

    if prediction[0] == 0:
        st.sidebar.success("🟢 Benign (No Cancer)")
    else:
        st.sidebar.error("🔴 Malignant (Cancer Detected)")



