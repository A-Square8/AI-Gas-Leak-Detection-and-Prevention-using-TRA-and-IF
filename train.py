import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, UpSampling1D, GaussianNoise, Dropout, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
import tensorflow as tf
import joblib

# Load preprocessed data
X_train_seq = np.load('data/X_train_seq.npy')
X_train_flat = np.load('data/X_train_flat.npy')

# Train Isolation Forest (initial rapid screening, as per literature)
if_model = IsolationForest(n_estimators=200, max_samples=0.8, contamination=0.1, bootstrap=True, random_state=42)
if_model.fit(X_train_flat)

# Build Robust LSTM Autoencoder (R-LSTM-AE: Twin for long-term dependencies, with noise and L1 reg)
def build_r_lstm_ae(input_shape):
    seq_len, features = input_shape  # Unpack for clarity: (10, 13)
    input_layer = Input(shape=input_shape)
    x = GaussianNoise(0.1)(input_layer)  # Add noise for robustness to sensor drift
    x = LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=l1(0.001))(x)
    x = Dropout(0.2)(x)
    x = LSTM(32, activation='relu', return_sequences=False, kernel_regularizer=l1(0.001))(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)  # Intermediate layer
    x = RepeatVector(seq_len)(x)  # Repeat across sequence length (10)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    output = Dense(features, activation='linear')(x)  # Match feature dimension (13)
    model = Model(input_layer, output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())  # Full loss function
    return model

input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
r_lstm_ae = build_r_lstm_ae(input_shape)
r_lstm_ae.fit(X_train_seq, X_train_seq, epochs=50, batch_size=256, validation_split=0.1, verbose=1)

# Build Robust 1D Convolutional Autoencoder (R-1D-CAE: Twin for short-term patterns, with noise and L1 reg)
def build_r_1d_cae(input_shape):
    input_layer = Input(shape=input_shape)  # e.g., (10, 13)
    x = GaussianNoise(0.1)(input_layer)

    # Encoder
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l1(0.001))(x)  # (10, 13) -> (10, 64)
    x = MaxPooling1D(pool_size=2, padding='same')(x)  # (10, 64) -> (5, 64)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l1(0.001))(x)  # (5, 64) -> (5, 32)
    encoded = MaxPooling1D(pool_size=2, padding='same')(x)  # (5, 32) -> (3, 32)

    # Decoder
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(encoded)  # (3, 32)
    x = UpSampling1D(size=2)(x)  # (3, 32) -> (6, 32)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)  # (6, 64)
    x = UpSampling1D(size=2)(x)  # (6, 64) -> (12, 64)
    x = tf.keras.layers.Cropping1D((1, 1))(x)  # Crop to (10, 64) to match input length
    output = Conv1D(input_shape[1], kernel_size=3, activation='linear', padding='same')(x)  # (10, 13)

    model = Model(input_layer, output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())  # Full loss function

    # Encoder model for hidden space
    encoder_model = Model(input_layer, encoded)
    return model, encoder_model

r_1d_cae, r_1d_cae_encoder = build_r_1d_cae(input_shape)
r_1d_cae.fit(X_train_seq, X_train_seq, epochs=50, batch_size=256, validation_split=0.1, verbose=1)

# Save models
joblib.dump(if_model, 'models/if_model.pkl')
r_lstm_ae.save('models/r_lstm_ae.h5')
r_1d_cae.save('models/r_1d_cae.h5')
r_1d_cae_encoder.save('models/r_1d_cae_encoder.h5')