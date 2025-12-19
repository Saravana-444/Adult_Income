import streamlit as st
import pickle
import numpy as np

st.title("Adult Income Prediction")

with open("adult_income_model.pkl", "rb") as f:
    model = pickle.load(f)

age = st.number_input("Age", 18, 100, 30)
education_num = st.number_input("Education Number", 1, 20, 10)
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
hours_per_week = st.number_input("Hours per Week", 1, 100, 40)

if st.button("Predict"):
    input_data = np.array([
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week
    ]).reshape(1, -1)

    prediction = model.predict(input_data)

    st.success("Income > 50K" if prediction[0] == 1 else "Income ≤ 50K")
