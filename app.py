
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

st.title("🎓 Student Performance Prediction")
st.write("Enter student information to predict the final mathematics grade (G3).")

@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")

model = load_model()

with st.form("student_form"):
    st.subheader("Student Information")

    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.slider("Age", 15, 22, 17)
    address = st.selectbox("Home address", ["U", "R"])
    famsize = st.selectbox("Family size", ["LE3", "GT3"])
    Pstatus = st.selectbox("Parents' cohabitation status", ["T", "A"])

    Medu = st.selectbox("Mother's education (0–4)", [0, 1, 2, 3, 4])
    Fedu = st.selectbox("Father's education (0–4)", [0, 1, 2, 3, 4])
    Mjob = st.selectbox("Mother's job", ["teacher", "health", "services", "at_home", "other"])
    Fjob = st.selectbox("Father's job", ["teacher", "health", "services", "at_home", "other"])
    reason = st.selectbox("School choice reason", ["home", "reputation", "course", "other"])
    guardian = st.selectbox("Guardian", ["mother", "father", "other"])

    traveltime = st.selectbox("Travel time (1–4)", [1, 2, 3, 4])
    studytime = st.selectbox("Weekly study time (1–4)", [1, 2, 3, 4])
    failures = st.selectbox("Past class failures", [0, 1, 2, 3])
    schoolsup = st.selectbox("Extra educational support", ["yes", "no"])
    famsup = st.selectbox("Family educational support", ["yes", "no"])
    paid = st.selectbox("Extra paid classes", ["yes", "no"])
    activities = st.selectbox("Extracurricular activities", ["yes", "no"])
    nursery = st.selectbox("Attended nursery school", ["yes", "no"])
    higher = st.selectbox("Wants higher education", ["yes", "no"])
    internet = st.selectbox("Internet access at home", ["yes", "no"])
    romantic = st.selectbox("In a romantic relationship", ["yes", "no"])

    famrel = st.slider("Family relationship quality (1–5)", 1, 5, 4)
    freetime = st.slider("Free time (1–5)", 1, 5, 3)
    goout = st.slider("Going out with friends (1–5)", 1, 5, 3)
    Dalc = st.slider("Workday alcohol consumption (1–5)", 1, 5, 1)
    Walc = st.slider("Weekend alcohol consumption (1–5)", 1, 5, 1)
    health = st.slider("Health status (1–5)", 1, 5, 3)
    absences = st.number_input("School absences", min_value=0, max_value=100, value=0)
    G1 = st.slider("First period grade (G1)", 0, 20, 10)
    G2 = st.slider("Second period grade (G2)", 0, 20, 10)

    submitted = st.form_submit_button("Predict Final Grade")

if submitted:
    data = pd.DataFrame([{
        "school": school, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
        "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences, "G1": G1, "G2": G2
    }])

    prediction = model.predict(data)[0]
    prediction = max(0, min(20, prediction))

    st.success(f"Predicted Final Grade (G3): {prediction:.2f} / 20")
    st.caption("This is a machine-learning estimate, not a guaranteed actual grade.")


