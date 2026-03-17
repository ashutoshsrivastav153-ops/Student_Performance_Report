import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.write("Developed by Ashutosh Srivastava")
st.title("🎯 Student Performance Predictor")

# Load dataset
df = pd.read_csv("C:\\Users\\DELL\\Desktop\\Student_Performance_Report\\student_data.csv")

# Features & Target
X = df[['study_hours','attendance','previous_marks']]
y = df['final_marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# User Input
st.header("Enter Student Details")

study_hours = st.slider("Study Hours", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 70)
previous_marks = st.slider("Previous Marks", 0, 100, 60)

# Prediction Button
if st.button("Predict"):
    input_data = np.array([[study_hours, attendance, previous_marks]])
    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Marks: {prediction[0]:.2f}")

    # Pass/Fail Logic
    if prediction[0] >= 40:
        st.success("✅ Status: Pass")
    else:
        st.error("❌ Status: Fail")