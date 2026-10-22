# Electricity Usage Anomaly Detector — BDA Mini Project

## What this project does
- Loads electricity consumption readings from a CSV file.
- Cleans invalid rows and sorts readings by timestamp.
- Flags unusual readings using either an IQR statistical rule or Isolation Forest.
- Shows summary metrics, time-series plot, usage histogram, flagged readings, and daily summary.
- Allows optional CSV upload and download of analyzed data.

**Important:** The included dataset is synthetic demonstration data. Flagged anomalies are not proof of electricity theft or a faulty meter.

## Setup (Windows)
1. Install Python 3.10+ from https://www.python.org/downloads/ . During setup, tick **Add Python to PATH**.
2. Extract this ZIP to a folder you can find easily, e.g. Desktop.
3. Open the extracted folder in VS Code.
4. In VS Code, choose **Terminal → New Terminal**.
5. Run:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```
6. A browser tab should open with the dashboard. If it does not, copy the Local URL shown in the terminal (usually http://localhost:8501) into your browser.

## If PowerShell blocks activation
You can skip activation and run:
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## CSV format for your own data
Required columns:
- `timestamp` (example: `2026-01-01 08:00:00`)
- `consumption_kwh` (numeric kWh reading)

Optional extra columns are preserved where possible.

## Suggested 3-day schedule
- Day 1: Setup, run the dashboard, understand the dataset.
- Day 2: Test IQR and Isolation Forest, capture screenshots, explain results.
- Day 3: Prepare report, PPT, and viva answers.

## Suggested report structure
1. Title page
2. Abstract
3. Problem statement
4. Objectives
5. Existing system and proposed system
6. Tools and technologies
7. Methodology / system architecture
8. Implementation and screenshots
9. Results and limitations
10. Conclusion and future scope
11. References

## Viva points
- BDA: extracting useful insights from datasets.
- Anomaly: a data point that differs substantially from the normal pattern.
- IQR: flags values outside Q1 - 1.5×IQR and Q3 + 1.5×IQR.
- Isolation Forest: an unsupervised algorithm that isolates unusual points.
- Streamlit: Python framework for interactive data apps.
- Limitation: unusual consumption needs investigation; it does not automatically prove the cause.
