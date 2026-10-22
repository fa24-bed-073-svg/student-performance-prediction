
import streamlit as st
import pandas as pd
import joblib

st.title("Student Performance Prediction")
st.write("Predict a student's final mathematics grade.")

model = joblib.load("student_performance_model.pkl")

st.info("Model loaded successfully!")
st.write("The prediction form will be added next.")
