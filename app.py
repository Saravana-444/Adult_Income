import streamlit as st
import pickle
import numpy as np

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Adult Income Census Prediction")
st.write("Predict whether a person's income is **>50K** or **≤50K**")

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    with open("Adult_Income.pkl", "rb") as file:
        df = pickle.load(file)
    return df

df = load_model()

# ---------------------------------
# Input Fields
# ---------------------------------
age = st.number_input("Age", min_value=17, max_value=100, step=1)

education_num = st.number_input(
    "Education Level (education-num)",
    min_value=1,
    max_value=16,
    step=1
)

capital_loss = st.number_input(
    "Capital Loss",
    min_value=0,
    step=100
)

hours_per_week = st.number_input(
    "Hours per Week",
    min_value=1,
    max_value=100,
    step=1
)

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Income"):
    try:
        input_data = np.array([[
            age,
            education_num,
            capital_loss,
            hours_per_week
        ]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("✅ Predicted Income: **> 50K**")
        else:
            st.success("✅ Predicted Income: **≤ 50K**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.write(e)
