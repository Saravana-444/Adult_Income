import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Adult Income Prediction", layout="centered")
st.title("💼 Adult Income Census Prediction")

@st.cache_resource
def load_model():
    with open("Adult_Income.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

age = st.number_input("Age", 17, 100, 30)
education_num = st.number_input("Education Number", 1, 16, 9)
capital_loss = st.number_input("Capital Loss", 0, 10000, 0)
hours_per_week = st.number_input("Hours per Week", 1, 100, 40)

if st.button("Predict Income"):
    input_data = np.array([[age, education_num, capital_loss, hours_per_week]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Income: >50K")
    else:
        st.success("✅ Income: ≤50K")
