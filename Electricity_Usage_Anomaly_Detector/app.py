import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Electricity Anomaly Detector", page_icon="⚡", layout="wide")
st.title("⚡ Electricity Usage Anomaly Detector")
st.caption("BDA Mini Project | Demonstration dataset uses synthetic readings; anomalies are indicators, not proof of meter faults or theft.")
from pathlib import Path
@st.cache_data
def load_default():
    return pd.read_csv(
        Path(__file__).parent / "electricity_readings.csv",
        parse_dates=["timestamp"]
    )

uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = load_default()
    except Exception:
        st.error("Could not find electricity_readings.csv. Keep it in the same folder as app.py.")
        st.stop()

required = {"timestamp", "consumption_kwh"}
if not required.issubset(df.columns):
    st.error("CSV must contain columns named: timestamp and consumption_kwh.")
    st.info("Example: timestamp, consumption_kwh")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["consumption_kwh"] = pd.to_numeric(df["consumption_kwh"], errors="coerce")
df = df.dropna(subset=["timestamp", "consumption_kwh"]).sort_values("timestamp").reset_index(drop=True)
if df.empty:
    st.error("No valid rows found in the uploaded CSV.")
    st.stop()

st.sidebar.header("Detection settings")
method = st.sidebar.selectbox("Detection method", ["IQR rule", "Isolation Forest"])
contamination = st.sidebar.slider("Expected anomaly fraction (Isolation Forest)", 0.01, 0.20, 0.05, 0.01)

if method == "IQR rule":
    q1 = df["consumption_kwh"].quantile(0.25)
    q3 = df["consumption_kwh"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["anomaly"] = (df["consumption_kwh"] < lower) | (df["consumption_kwh"] > upper)
else:
    model = IsolationForest(contamination=contamination, random_state=42)
    features = df[["consumption_kwh"]].copy()
    df["anomaly"] = model.fit_predict(features) == -1

total = len(df)
anomalies = int(df["anomaly"].sum())
avg = df["consumption_kwh"].mean()
peak = df["consumption_kwh"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Valid readings", f"{total:,}")
c2.metric("Flagged readings", f"{anomalies:,}")
c3.metric("Average usage", f"{avg:.2f} kWh")
c4.metric("Peak usage", f"{peak:.2f} kWh")

st.subheader("Consumption over time")
fig, ax = plt.subplots(figsize=(12, 4))
normal = df[~df["anomaly"]]
odd = df[df["anomaly"]]
ax.plot(df["timestamp"], df["consumption_kwh"], linewidth=1.2, label="Consumption")
if not odd.empty:
    ax.scatter(odd["timestamp"], odd["consumption_kwh"], marker="x", s=55, label="Flagged anomaly")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Consumption (kWh)")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

left, right = st.columns(2)
with left:
    st.subheader("Usage distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.hist(df["consumption_kwh"], bins=20)
    ax2.set_xlabel("Consumption (kWh)")
    ax2.set_ylabel("Number of readings")
    ax2.grid(True, alpha=0.2)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

with right:
    st.subheader("Flagged readings")
    flagged = df[df["anomaly"]].copy()
    st.dataframe(flagged, use_container_width=True, hide_index=True)

st.subheader("Daily summary")
daily = df.set_index("timestamp").resample("D")["consumption_kwh"].agg(["mean", "max", "sum"]).reset_index()
daily.columns = ["Date", "Average kWh", "Peak kWh", "Total kWh"]
st.dataframe(daily, use_container_width=True, hide_index=True)

csv_out = df.to_csv(index=False).encode("utf-8")
st.download_button("Download analyzed readings (CSV)", data=csv_out, file_name="analyzed_electricity_readings.csv", mime="text/csv")

st.markdown("---")
st.caption("Interpretation: an anomaly is a reading that differs from the dataset's pattern. Check the meter, appliance usage, and context before drawing real-world conclusions.")
