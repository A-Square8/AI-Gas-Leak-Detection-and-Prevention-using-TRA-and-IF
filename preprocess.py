import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('data/gas_sensor_complete.csv')

# Drop unnecessary columns
df = df.drop(['batch_id', 'concentration_range'], axis=1)

# Encode gas_name for per-gas evaluation
le = LabelEncoder()
df['gas_label'] = le.fit_transform(df['gas_name'])
gas_names = le.classes_

# Features: feature_001 to feature_128
features = [col for col in df.columns if 'feature_' in col]
X = df[features].values
y_gas = df['gas_label'].values

# Handle NaNs
X = np.nan_to_num(X)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction (retain 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Create sliding window sequences (for time-series models like LSTM/Conv1D)
def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

seq_len = 10
X_seq = create_sequences(X_pca, seq_len)
y_gas_seq = y_gas[seq_len-1:]  # Align labels with sequences

# Split: Train on normal, test will have anomalies injected
X_train_seq, X_test_seq, y_gas_train, y_gas_test = train_test_split(X_seq, y_gas_seq, test_size=0.2, stratify=y_gas_seq, random_state=42)

# Flatten for IF (non-sequential model)
X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
X_test_flat_normal = X_test_seq.reshape(X_test_seq.shape[0], -1)

# Inject anomalies into test set (20% anomalies: 50% drift noise, 50% spikes)
def inject_anomalies(X_seq, X_flat, anomaly_ratio=0.2, noise_level=3.0, spike_level=5.0):
    n_anomalies = int(len(X_seq) * anomaly_ratio)
    anomalies_idx = np.random.choice(len(X_seq), n_anomalies, replace=False)

    X_seq_anom = X_seq.copy()
    X_flat_anom = X_flat.copy()
    for idx in anomalies_idx:
        if np.random.rand() < 0.5:  # Gradual drift
            noise = np.random.normal(0, noise_level, X_seq[idx].shape)
            X_seq_anom[idx] += noise
        else:  # Sudden spike
            X_seq_anom[idx] *= np.random.uniform(1.5, spike_level)
    X_flat_anom = X_seq_anom.reshape(X_seq_anom.shape[0], -1)

    y_anom = np.zeros(len(X_seq))
    y_anom[anomalies_idx] = 1
    return X_seq_anom, X_flat_anom, y_anom

X_test_seq, X_test_flat, y_true = inject_anomalies(X_test_seq, X_test_flat_normal)

# Save preprocessed data
np.save('data/X_train_seq.npy', X_train_seq)
np.save('data/X_train_flat.npy', X_train_flat)
np.save('data/X_test_seq.npy', X_test_seq)
np.save('data/X_test_flat.npy', X_test_flat)
np.save('data/y_gas_train.npy', y_gas_train)
np.save('data/y_gas_test.npy', y_gas_test)
np.save('data/y_true.npy', y_true)
np.save('data/gas_names.npy', gas_names)