import streamlit as st
import pandas as pd, numpy as np, joblib, os

st.set_page_config(page_title="Diabetes Readmission Predictor", page_icon="🏥", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join("sample_data", "gb_best_model.joblib"))
    pre = joblib.load(os.path.join("sample_data", "preprocessor.joblib"))
    return model, pre

model, pre = load_assets()
THRESHOLD = 0.45

st.title("🏥 Diabetes Readmission Predictor")
st.write("Predict if a diabetic patient will be readmitted within 30 days after discharge.")

with st.sidebar:
    st.header("Patient Info")
    race = st.selectbox("Race", ["Caucasian","AfricanAmerican","Hispanic","Asian","Other"])
    gender = st.selectbox("Gender", ["Male","Female"])
    age_group = st.selectbox("Age Group", ["Young","Middle","Elderly"])
    time_in_hospital = st.number_input("Days in Hospital", 1, 14, 4)
    num_lab_procedures = st.number_input("Lab Procedures", 1, 100, 40)
    num_medications = st.number_input("Medications", 1, 50, 10)
    number_outpatient = st.number_input("Outpatient Visits", 0, 20, 0)
    number_emergency = st.number_input("Emergency Visits", 0, 20, 0)
    number_inpatient = st.number_input("Inpatient Visits", 0, 20, 0)
    total_visits = number_outpatient + number_emergency + number_inpatient
    admission_type_id = st.selectbox("Admission Type ID", [1,2,3,4,5,6,7,8])
    discharge_disposition_id = st.selectbox("Discharge Disposition ID", [1,2,3,4,5,6,7,8,9,10])
    admission_source_id = st.selectbox("Admission Source ID", [1,2,3,4,5,6,7,8,9,10])

    predict = st.button("🔍 Predict Readmission")

if predict:
    sample = {
        "race": race, "gender": gender, "age_group": age_group,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "total_visits": total_visits,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id
    }

    df = pd.DataFrame([sample])
    for col in pre.feature_names_in_:
        if col not in df.columns:
            df[col] = np.nan
    df = df[pre.feature_names_in_]

    Xp = pre.transform(df)
    prob = model.predict_proba(Xp)[0,1]
    pred = int(prob >= THRESHOLD)

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"High Risk of Readmission (Prob = {prob:.1%})")
    else:
        st.success(f"Low Risk of Readmission (Prob = {prob:.1%})")
    st.progress(float(prob))
