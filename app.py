# Dashborad implementation of the application will be done here
# Dashboard shall contain the following:
# Left Side :
# (1)Live reading received from the sensors alomgwith a time vs concentration graph for each of the gas type

# Right Side:
# (1)Historical data analysis for each of the gas type
# (2)Anomaly detection status (Normal/Anomalous) for each of the gas type
# (3)Overall system status (Normal/Anomalous)
# (4)Confusion matrix for the model performance
# (5)Precision, Recall, F1-score for the model performance
# (6)ROC-AUC curve for the model performance
# (7)Threshold value used for anomaly detection
# (8)Gas type wise performance metrics



# Dashboard shall be implemented using  Streamlit
# It shall use testing data from ('data/') only
# IT shall be implemented in such a way that it receive 400 data rows per minute as there will be 2400 rows in testing field(so it can be done in 6 minutes)


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# Load preprocessed test data
X_test_seq = np.load('data/X_test_seq.npy')
X_test_flat = np.load('data/X_test_flat.npy')
y_true = np.load('data/y_true.npy')
y_gas_test = np.load('data/y_gas_test.npy')
gas_names = np.load('data/gas_names.npy', allow_pickle=True)

# Load or simulate hybrid_scores (replace with actual loading from test.py)
hybrid_scores = np.load('data/hybrid_scores.npy')  # Ensure this file is saved in test.py

# Function to compute predictions
threshold = np.percentile(hybrid_scores, 80)
y_pred = (hybrid_scores > threshold).astype(int)

# Streamlit app layout
st.title("Gas Sensor Anomaly Detection Dashboard")

# Left Side: Live Readings and Graphs
st.sidebar.header("Live Sensor Readings")

# Simulate streaming data without blocking
total_rows = len(X_test_seq)
rows_per_minute = 400
data_placeholder = st.sidebar.empty()
time_placeholder = st.sidebar.empty()
fig_placeholder = st.sidebar.empty()

current_time = datetime.now()
data_stream = []

for i in range(0, total_rows, rows_per_minute):
    batch = X_test_seq[i:min(i + rows_per_minute, total_rows)]
    data_stream.extend(batch)
    times = [current_time + timedelta(seconds=j) for j in range(len(data_stream))]
    data_placeholder.text(f"Data Rows Processed: {len(data_stream)}/{total_rows}")
    time_placeholder.text(f"Current Time: {times[-1].strftime('%H:%M:%S')}")
    
    # Update graph
    fig, ax = plt.subplots()
    for gas_idx, gas_name in enumerate(gas_names):
        mask = (y_gas_test[:len(data_stream)] == gas_idx)
        ax.plot(times[:len(data_stream)][mask], [d[0, 0] for d in data_stream][:len(data_stream)][mask], label=gas_name)
    ax.legend()
    fig_placeholder.pyplot(fig)
    
    time.sleep(1)  # Update every second (adjust as needed)

# Right Side: Historical Data and Metrics
st.header("Historical Data Analysis & Metrics")

# (1) Historical Data Analysis
st.subheader("Historical Data per Gas Type")
for gas_idx, gas_name in enumerate(gas_names):
    mask = (y_gas_test == gas_idx)
    st.write(f"{gas_name}: {mask.sum()} samples")

# (2) Anomaly Detection Status
st.subheader("Anomaly Detection Status")
for gas_idx, gas_name in enumerate(gas_names):
    mask = (y_gas_test == gas_idx)
    status = "Anomalous" if y_pred[mask].any() else "Normal"
    st.write(f"{gas_name}: {status}")

# (3) Overall System Status
overall_status = "Anomalous" if y_pred.any() else "Normal"
st.write(f"Overall System Status: {overall_status}")

# (4) Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
st.write(cm)

# (5) Precision, Recall, F1-score
st.subheader("Model Performance Metrics")
prec = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
st.write(f"Precision: {prec:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# (6) ROC-AUC Curve
st.subheader("ROC-AUC Curve")
fpr, tpr, _ = roc_curve(y_true, hybrid_scores)
roc_auc = roc_auc_score(y_true, hybrid_scores)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.legend()
st.pyplot(fig)

# (7) Threshold Value
st.subheader("Anomaly Detection Threshold")
st.write(f"Threshold: {threshold:.4f}")

# (8) Gas Type Wise Performance Metrics
st.subheader("Gas Type Wise Performance")
for gas_idx, gas_name in enumerate(gas_names):
    mask = (y_gas_test == gas_idx)
    if mask.sum() > 0:
        prec_gas = precision_score(y_true[mask], y_pred[mask])
        recall_gas = recall_score(y_true[mask], y_pred[mask])
        f1_gas = f1_score(y_true[mask], y_pred[mask])
        st.write(f"{gas_name}: Precision: {prec_gas:.4f}, Recall: {recall_gas:.4f}, F1-score: {f1_gas:.4f}")