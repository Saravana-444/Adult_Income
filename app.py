import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Adult Income Prediction")

st.title("💼 Adult Income Prediction")

# Load trained model
with open("Adult_Income.pkl", "rb") as f:
    df = pickle.load(f)

# User inputs (ONLY NUMBERS)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
education_num = st.number_input("Education Number", min_value=1, max_value=20, value=10)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

# Prediction
if st.button("Predict Income"):
    input_data = np.array([
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week
    ]).reshape(1, -1)

    prediction = df.predict(input_data)

    if prediction[0] == 1:
        st.success("💰 Income > 50K")
    else:
        st.warning("💵 Income ≤ 50K")
