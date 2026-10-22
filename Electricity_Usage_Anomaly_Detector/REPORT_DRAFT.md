# Electricity Usage Anomaly Detector
## BDA Mini Project — Draft Report

### Abstract
This project demonstrates a data analytics application that examines electricity consumption readings and flags values that differ from the overall pattern. The application uses Python-based data processing and anomaly-detection methods and presents results through an interactive dashboard. The bundled dataset is synthetic and intended for demonstration.

### Problem Statement
Large sets of electricity readings can be difficult to inspect manually. Unusual readings may be overlooked when data volume grows. A simple analytics dashboard can help users identify readings that deserve further investigation.

### Objectives
1. Load and clean electricity usage readings.
2. Calculate summary statistics.
3. Detect unusual readings using statistical and machine-learning methods.
4. Visualize consumption trends and flagged observations.
5. Export analyzed results for further review.

### Tools and Technologies
- Python
- Pandas and NumPy
- Scikit-learn (Isolation Forest)
- Matplotlib
- Streamlit
- CSV dataset

### Methodology
1. Read timestamped consumption data from CSV.
2. Convert timestamps and consumption values to valid data types.
3. Remove rows with invalid required values.
4. Apply either the Interquartile Range (IQR) rule or Isolation Forest.
5. Display metrics, time-series plot, histogram, flagged rows, and daily summaries.

### Results
The dashboard calculates the number of valid readings, number of flagged readings, average consumption, and peak consumption. The exact results depend on the selected detection method and dataset. Add screenshots and values from your actual run here.

### Limitations
The included dataset is synthetic. An anomaly is only a statistical flag and does not establish the cause. Real deployment would require validated meter data, context about occupancy and appliances, and careful evaluation.

### Conclusion and Future Scope
The prototype demonstrates an end-to-end analytics workflow from CSV ingestion to anomaly detection and visualization. Future improvements could include real smart-meter data, time-of-day baselines, alerts, and model evaluation.

### References
- Pandas documentation: https://pandas.pydata.org/docs/
- Streamlit documentation: https://docs.streamlit.io/
- Scikit-learn Isolation Forest documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
