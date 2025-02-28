import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("CreditRiskModel").getOrCreate()

model = PipelineModel.load("path/to/your/model")  #enter path

def predict(input_data):
    input_df = spark.createDataFrame([input_data])
    prediction = model.transform(input_df)
    return prediction.select("prediction", "probability").collect()[0]

st.set_page_config(page_title="Credit Risk Modeling", page_icon="📊", layout="centered")
st.title("📊 Customer Credit Risk Modeling")

st.subheader("💼 Customer Details")

col1, col2, col3 = st.columns(3)
age = col1.number_input("Age", min_value=18, max_value=100, value=28)
income = col2.number_input("Income (Annual)", min_value=0, max_value=5000000, value=290875)
loan_amount = col3.number_input("Loan Amount", min_value=0, value=2560000)

st.subheader("📊 Loan Insights")
lti = loan_amount / income if income > 0 else 0
st.metric(label="Loan-to-Income Ratio (LTI)", value=f"{lti:.2f}")

loan_purpose = st.selectbox("Loan Purpose", ['Education', 'Home', 'Auto', 'Personal'])
loan_type = st.radio("Loan Type", ['Unsecured', 'Secured'])

if st.button("Calculate Risk"):
    input_data = {
        "loan_amnt": loan_amount,
        "annual_inc": income,
        "dti": 0,  # 
        "home_ownership": 1,  
        "purpose": loan_purpose
    }
    
    prediction = predict(input_data)
    st.success("✅ Risk Assessment Completed!")
    st.write(f"**Default Probability:** {prediction['probability'][1]:.2%}")
    st.write(f"**Credit Score:** {prediction['prediction']}")

spark.stop()
