
from pathlib import Path
import re

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from pypdf import PdfReader
from PIL import Image
import pytesseract


# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Electricity Usage Anomaly Detector",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electricity Usage Anomaly Detector")
st.caption(
    "BDA Mini Project | Bill details are estimates. "
    "Always verify extracted values against the original bill."
)


# ---------------- Electricity Bill Upload ----------------
st.sidebar.header("Upload Electricity Bill")

bill_file = st.sidebar.file_uploader(
    "Upload bill PDF or photo (JPG/PNG)",
    type=["pdf", "png", "jpg", "jpeg"],
    key="bill_upload"
)


def extract_bill_text(uploaded_file):
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]
        return "\n".join(pages), "PDF text extraction"

    image = Image.open(uploaded_file).convert("RGB")
    return pytesseract.image_to_string(image), "OCR from bill image"


def first_match(patterns, text, flags=re.IGNORECASE):
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1).strip()
    return None


if bill_file is not None:
    st.subheader("📄 Uploaded Bill Summary")

    try:
        bill_text, extraction_method = extract_bill_text(bill_file)

        if not bill_text.strip():
            st.warning(
                "No readable text found. If this is a scanned PDF, "
                "try uploading a clear JPG or PNG bill photo."
            )

        else:
            units = first_match(
                [
                    r"(?:units?\s+consumed|energy\s+consumed|"
                    r"consumption|total\s+units|kwh\s+consumed)"
                    r"\s*[:\-]?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units|kwh)\b"
                ],
                bill_text
            )

            amount = first_match(
                [
                    r"(?:total\s+amount\s+due|amount\s+payable|"
                    r"net\s+amount\s+payable|bill\s+amount|"
                    r"total\s+bill|amount\s+due)"
                    r"\s*[:\-]?\s*(?:₹|rs\.?|inr)?\s*"
                    r"([0-9,]+(?:\.[0-9]{1,2})?)",
                    r"(?:₹|rs\.?|inr)\s*"
                    r"([0-9,]+(?:\.[0-9]{1,2})?)"
                ],
                bill_text
            )

            bill_date = first_match(
                [
                    r"(?:bill\s+date|date\s+of\s+issue|issue\s+date)"
                    r"\s*[:\-]?\s*([0-3]?\d[\/\-][01]?\d[\/\-](?:20)?\d{2})",
                    r"(?:bill\s+date|date\s+of\s+issue|issue\s+date)"
                    r"\s*[:\-]?\s*(\d{4}[\/\-][01]?\d[\/\-][0-3]?\d)"
                ],
                bill_text
            )

            period = first_match(
                [
                    r"(?:billing\s+period|bill\s+period|period)"
                    r"\s*[:\-]?\s*([^\n]{3,45})"
                ],
                bill_text
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Units Consumed (candidate)",
                f"{units} kWh" if units else "Not detected"
            )

            c2.metric(
                "Bill Amount (candidate)",
                f"₹{amount}" if amount else "Not detected"
            )

            c3.metric(
                "Bill Date (candidate)",
                bill_date or "Not detected"
            )

            st.write(
                "**Billing Period (candidate):**",
                period or "Not detected"
            )

            st.caption(
                f"Extraction method: {extraction_method}. "
                "Bill formats differ, so verify every value."
            )

            with st.expander("View extracted text to verify"):
                st.text(bill_text[:12000])

    except Exception as e:
        st.error(f"Could not read this bill: {e}")
        st.info(
            "Try a text-based PDF or a clear JPG/PNG photo. "
            "OCR for bill photos requires Tesseract to be installed."
        )


st.divider()


# ---------------- Readings CSV Upload ----------------
st.sidebar.header("Usage Data for Anomaly Detection")

uploaded_csv = st.sidebar.file_uploader(
    "Upload readings CSV (optional)",
    type=["csv"],
    key="readings_csv"
)


@st.cache_data
def load_default():
    csv_path = Path(__file__).parent / "electricity_readings.csv"
    return pd.read_csv(csv_path, parse_dates=["timestamp"])


if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()

else:
    try:
        df = load_default()

    except FileNotFoundError:
        st.warning(
            "Default electricity_readings.csv was not found. "
            "Upload a readings CSV to use anomaly detection."
        )
        st.stop()

    except Exception as e:
        st.error(f"Could not load default CSV: {e}")
        st.stop()


# ---------------- Validate and Clean Data ----------------
required_columns = {"timestamp", "consumption_kwh"}

if not required_columns.issubset(df.columns):
    st.error(
        "Readings CSV must contain columns named "
        "timestamp and consumption_kwh."
    )
    st.info(
        "Example headers: timestamp, consumption_kwh"
    )
    st.stop()


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["consumption_kwh"] = pd.to_numeric(
    df["consumption_kwh"],
    errors="coerce"
)

df = (
    df.dropna(subset=["timestamp", "consumption_kwh"])
    .sort_values("timestamp")
    .reset_index(drop=True)
)

if df.empty:
    st.error("No valid rows found in the readings CSV.")
    st.stop()


# ---------------- Detection Settings ----------------
st.sidebar.subheader("Detection Settings")

method = st.sidebar.selectbox(
    "Detection Method",
    ["IQR Rule", "Isolation Forest"]
)

contamination = st.sidebar.slider(
    "Expected anomaly fraction (Isolation Forest)",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01
)


# ---------------- Detect Anomalies ----------------
if method == "IQR Rule":
    q1, q3 = df["consumption_kwh"].quantile([0.25, 0.75])
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    df["anomaly"] = (
        (df["consumption_kwh"] < lower_limit)
        | (df["consumption_kwh"] > upper_limit)
    )

else:
    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    df["anomaly"] = (
        model.fit_predict(df[["consumption_kwh"]]) == -1
    )


flagged = df[df["anomaly"]]


# ---------------- Usage Overview ----------------
st.subheader("Usage Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Valid Readings", f"{len(df):,}")
c2.metric("Flagged Readings", f"{int(df['anomaly'].sum()):,}")
c3.metric(
    "Average Usage",
    f"{df['consumption_kwh'].mean():.2f} kWh"
)
c4.metric(
    "Peak Usage",
    f"{df['consumption_kwh'].max():.2f} kWh"
)


# ---------------- Consumption Graph ----------------
st.subheader("Electricity Consumption Over Time")

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    df["timestamp"],
    df["consumption_kwh"],
    linewidth=1.2,
    label="Consumption"
)

if not flagged.empty:
    ax.scatter(
        flagged["timestamp"],
        flagged["consumption_kwh"],
        marker="x",
        s=55,
        label="Flagged Anomaly"
    )

ax