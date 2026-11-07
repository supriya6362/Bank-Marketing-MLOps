import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Bank Marketing MLOps", layout="wide")

st.title("Bank Marketing MLOps Dashboard")
st.subheader("Lab 12 - Monitoring and Lab 13 - Continuous Prediction")

output = Path("outputs")

col1, col2 = st.columns(2)

with col1:
    st.metric("Lab 12", "Model Monitoring")

with col2:
    st.metric("Lab 13", "Prediction Monitoring")

st.divider()

st.header("Lab 13 - Continuous Prediction Monitoring")

csv_file = output / "lab13_prediction_log.csv"
json_file = output / "lab13_monitoring_summary.json"

if json_file.exists():
    with open(json_file) as f:
        st.json(json.load(f))
else:
    st.info("Lab 13 summary not found.")

if csv_file.exists():
    df = pd.read_csv(csv_file)
    st.write("Prediction Records:", len(df))
    st.dataframe(df, use_container_width=True)

    numeric = df.select_dtypes(include="number").columns.tolist()
    if numeric:
        column = st.selectbox("Select column for chart", numeric)
        st.line_chart(df[column])
else:
    st.info("Lab 13 prediction log not found.")

st.divider()

st.header("Lab 12 - Monitoring Reports")

if output.exists():
    reports = list(output.rglob("*.csv"))

    for report in reports:
        st.subheader(report.name)
        try:
            data = pd.read_csv(report)
            st.dataframe(data, use_container_width=True)
        except Exception as e:
            st.error(str(e))
else:
    st.warning("Outputs folder not found.")

st.success("Bank Marketing MLOps Dashboard is running!")