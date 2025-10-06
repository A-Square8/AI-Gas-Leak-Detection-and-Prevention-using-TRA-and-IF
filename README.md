#Real Time Gas Sensor Data Analysis Using TRA and IF

This project focuses on analyzing gas sensor data using various machine learning algorithms. The goal is to preprocess the data, train models, and evaluate their performance for gas classification and anomaly detection.


---

## Dataset

- **Source:** [UCI Machine Learning Repository - Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset)
- **Files Used:**
  - `data/gas_sensor_complete.csv`
  - Preprocessed numpy arrays: `X_train_flat.npy`, `X_train_seq.npy`, `X_test_flat.npy`, `X_test_seq.npy`, `y_gas_train.npy`, `y_gas_test.npy`, `y_true.npy`, `gas_names.npy`

## Project Structure

```
.
├── app.py
├── preprocess.py
├── requirements.txt
├── test.py
├── train.py
├── data/
│   ├── gas_names.npy
│   ├── gas_sensor_complete.csv
│   ├── X_test_flat.npy
│   ├── X_test_seq.npy
│   ├── X_train_flat.npy
│   ├── X_train_seq.npy
│   ├── y_gas_test.npy
│   ├── y_gas_train.npy
│   └── y_true.npy
├── models/
│   ├── if_model.pkl
│   ├── r_1d_cae_encoder.h5
│   ├── r_1d_cae.h5
│   └── r_lstm_ae.h5
└── .gitignore
```

## Requirements

Install dependencies using:

```sh
pip install -r requirements.txt
```

## Preprocessing

- Data is loaded and preprocessed using [`preprocess.py`](preprocess.py).
- Preprocessing steps include normalization, reshaping, and splitting into train/test sets.
- Preprocessed data is saved as `.npy` files in the `data/` directory.

## Machine Learning Algorithms Used

- **Isolation Forest** (`if_model.pkl`): For anomaly detection.
- **1D Convolutional Autoencoder** (`r_1d_cae.h5`, `r_1d_cae_encoder.h5`): For feature extraction and anomaly detection.
- **LSTM Autoencoder** (`r_lstm_ae.h5`): For sequence modeling and anomaly detection.

## Training

- Models are trained using [`train.py`](train.py).
- Trained models are saved in the `models/` directory.

## Evaluation & Testing

- Model evaluation and testing are performed using [`test.py`](test.py).
- Results are compared using metrics such as accuracy, precision, recall, and F1-score.

## Usage

1. **Preprocess the data:**
   ```sh
   python preprocess.py
   ```
2. **Train the models:**
   ```sh
   python train.py
   ```
3. **Test/Evaluate the models:**
   ```sh
   python test.py
   ```
4. **Run the application (if applicable):**
   ```sh
   python app.py
   ```

## Results

- Results and performance metrics will be displayed after running `test.py`.
- Model files are stored in the `models/` directory.

## References

- [Gas Sensor Array Drift Dataset at UCI](https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset)
- [Isolation Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Keras Autoencoder Examples](https://keras.io/examples/)

---

