import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =====================================================
# 🏥 Diabetes Readmission Predictor — Streamlit App
# =====================================================

st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 🔧 Load Model and Preprocessor Safely
# =====================================================

@st.cache_resource
def load_assets():
    try:
        base_path = os.path.dirname(__file__)  # Folder where this file is
        model_path = os.path.join(base_path, "sample_data", "gb_best_model.joblib")
        pre_path = os.path.join(base_path, "sample_data", "preprocessor.joblib")

        model = joblib.load(model_path)
        pre = joblib.load(pre_path)
        return model, pre

    except FileNotFoundError as e:
        st.error("⚠️ Model or preprocessor file not found. Please verify the 'sample_data' folder exists.")
        st.stop()

model, pre = load_assets()
THRESHOLD = 0.45

# =====================================================
# 💡 App Layout
# =====================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f9ff;
    }
    .stButton>button {
        color: white;
        background-color: #0077b6;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0096c7;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏥 Diabetes Readmission Predictor")
st.markdown(
    """
    Use this app to estimate whether a diabetic patient is **likely to be readmitted within 30 days** after discharge.
    The prediction is powered by a tuned **Gradient Boosting model** trained on real hospital data.
    """
)

# =====================================================
# 🧾 Sidebar Inputs
# =====================================================
with st.sidebar:
    st.header("🧍 Patient Information")

    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age_group = st.selectbox("Age Group", ["Young", "Middle", "Elderly"])
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

# =====================================================
# 🤖 Prediction Section
# =====================================================
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

    # Align columns with training features
    for col in pre.feature_names_in_:
        if col not in df.columns:
            df[col] = np.nan
    df = df[pre.feature_names_in_]

    # Predict
    Xp = pre.transform(df)
    prob = model.predict_proba(Xp)[0, 1]
    pred = int(prob >= THRESHOLD)

    # =====================================================
    # 🎯 Display Results
    # =====================================================
    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error(f"🟥 **High Risk of Readmission** — Probability: {prob:.1%}")
    else:
        st.success(f"🟩 **Low Risk of Readmission** — Probability: {prob:.1%}")

    st.progress(float(prob))
    st.caption(f"Threshold used for decision: {THRESHOLD:.2f}")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Gradient Boosting (2025).")
