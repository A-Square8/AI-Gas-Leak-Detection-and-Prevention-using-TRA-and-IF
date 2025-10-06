import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf
# Load preprocessed test data
X_test_seq = np.load('data/X_test_seq.npy')
X_test_flat = np.load('data/X_test_flat.npy')
y_true = np.load('data/y_true.npy')
y_gas_test = np.load('data/y_gas_test.npy')
gas_names = np.load('data/gas_names.npy', allow_pickle=True)

# Define custom objects to resolve 'mse'
custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}

# Load trained models with custom objects
if_model = joblib.load('models/if_model.pkl')
r_lstm_ae = load_model('models/r_lstm_ae.h5', custom_objects=custom_objects)
r_1d_cae = load_model('models/r_1d_cae.h5', custom_objects=custom_objects)
r_1d_cae_encoder = load_model('models/r_1d_cae_encoder.h5', custom_objects=custom_objects)

# Predict with Isolation Forest
if_scores = -if_model.decision_function(X_test_flat)  # Higher scores indicate anomalies

# Reconstruction errors for R-LSTM-AE (input space)
lstm_recon = r_lstm_ae.predict(X_test_seq)
lstm_errors = np.mean((X_test_seq - lstm_recon) ** 2, axis=(1, 2))

# Reconstruction errors for R-1D-CAE (input + hidden space)
cae_recon = r_1d_cae.predict(X_test_seq)
cae_input_errors = np.mean((X_test_seq - cae_recon) ** 2, axis=(1, 2))
cae_hidden = r_1d_cae_encoder.predict(X_test_seq)
cae_hidden_recon = r_1d_cae_encoder.predict(cae_recon)  # Approximate hidden reconstruction
cae_hidden_errors = np.mean((cae_hidden - cae_hidden_recon) ** 2, axis=(1, 2))
cae_errors = (cae_input_errors + cae_hidden_errors) / 2

# Twin AE scores (average of R-LSTM-AE and R-1D-CAE)
twin_ae_scores = (lstm_errors + cae_errors) / 2

# Hybrid scores: Weighted ensemble (60% Twin AE + 40% IF)
hybrid_scores = 0.6 * twin_ae_scores + 0.4 * if_scores

# Threshold: Use percentile for unsupervised (top 20% as anomalies, matching injection ratio)
threshold = np.percentile(hybrid_scores, 80)  # Adjust based on anomaly ratio (e.g., 20% anomalies)
y_pred = (hybrid_scores > threshold).astype(int)

# Evaluate with confusion matrix
def evaluate_with_confusion(y_true, y_pred, y_gas, gas_names):
    # Overall metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()  # TN, FP, FN, TP
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Overall - Acc: {acc:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}")
    print(f"Overall Confusion Matrix - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # Per-gas metrics
    for gas_idx, gas_name in enumerate(gas_names):
        mask = (y_gas == gas_idx)
        if mask.sum() > 0:
            y_true_gas = y_true[mask]
            y_pred_gas = y_pred[mask]
            cm_gas = confusion_matrix(y_true_gas, y_pred_gas)
            tn_gas, fp_gas, fn_gas, tp_gas = cm_gas.ravel()
            acc_gas = accuracy_score(y_true_gas, y_pred_gas)
            prec_gas = precision_score(y_true_gas, y_pred_gas, zero_division=0)
            f1_gas = f1_score(y_true_gas, y_pred_gas, zero_division=0)
            print(f"{gas_name} - Acc: {acc_gas:.4f}, Prec: {prec_gas:.4f}, F1: {f1_gas:.4f}")
            print(f"{gas_name} Confusion Matrix - TP: {tp_gas}, TN: {tn_gas}, FP: {fp_gas}, FN: {fn_gas}")

evaluate_with_confusion(y_true, y_pred, y_gas_test, gas_names)

