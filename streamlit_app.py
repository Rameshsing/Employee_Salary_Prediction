# streamlit_income_app.py

import streamlit as st
import pandas as pd
import joblib

# Load saved artifacts
model = joblib.load('models/income_rf_model.pkl')
scaler = joblib.load('models/income_scaler.pkl')
label_encoders = joblib.load('models/income_label_encoders.pkl')

st.title("💼 Employee Monthly Income Prediction")
st.markdown("""
Predict an employee’s **Monthly Income** based on key factors:
- OverTime  
- Age  
- Total Working Years  
- Stock Option Level  
- Work–Life Balance  
- Job Role  
- Education Field  
""")

with st.form("input_form"):
    # 1) OverTime
    ot = st.selectbox("Over Time?", label_encoders['OverTime'].classes_)
    # 2) Age
    age = st.slider("Age", min_value=18, max_value=60, value=30)
    # 3) Total Working Years
    twy = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
    # 4) Stock Option Level
    sol = st.selectbox("Stock Option Level", (0, 1, 2, 3))
    # 5) Work–Life Balance
    wlb = st.selectbox("Work–Life Balance Rating", (1, 2, 3, 4))
    # 6) Job Role
    jr = st.selectbox("Job Role", label_encoders['JobRole'].classes_)
    # 7) Education Field
    ef = st.selectbox("Education Field", label_encoders['EducationField'].classes_)
    submit = st.form_submit_button("🔮 Predict Monthly Income")

if submit:
    # Prepare DataFrame for prediction
    df_input = pd.DataFrame([{
        'OverTime': ot,
        'Age': age,
        'TotalWorkingYears': twy,
        'StockOptionLevel': sol,
        'WorkLifeBalance': wlb,
        'JobRole': jr,
        'EducationField': ef
    }])

    # Encode categorical inputs
    for col in ['OverTime', 'JobRole', 'EducationField']:
        df_input[col] = label_encoders[col].transform(df_input[col].astype(str))

    # Scale numerical inputs
    df_input[['Age', 'TotalWorkingYears', 'StockOptionLevel', 'WorkLifeBalance']] = scaler.transform(
        df_input[['Age', 'TotalWorkingYears', 'StockOptionLevel', 'WorkLifeBalance']]
    )

    # Predict and display result
    income_pred = model.predict(df_input)[0]
    st.success(f"Estimated Monthly Income: **${income_pred:,.0f}**")
