import streamlit as st
import pickle
import numpy as np

# Load files
with open("adult_income_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Adult Income Prediction")

age = st.number_input("Age", 18, 100)
education = st.selectbox("Education", ["Bachelors", "HS-grad", "Masters"])
occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Sales"])
hours = st.number_input("Hours per week", 1, 100)

if st.button("Predict Income"):
    cat_data = encoder.transform([[education, occupation]])
    num_data = scaler.transform([[age, hours]])

    input_data = np.hstack((num_data, cat_data))

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Income > 50K")
    else:
        st.warning("Income ≤ 50K")
