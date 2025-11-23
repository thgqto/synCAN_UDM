import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import zipfile
import io
import joblib

# Step 1: Load and concat (same)
train_files = ['train_1.zip', 'train_2.zip', 'train_3.zip', 'train_4.zip']
dfs = []
for zip_path in train_files:
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_name = [name for name in z.namelist() if name.lower().endswith('.csv')][0]
        with z.open(csv_name) as f:
            df = pd.read_csv(io.StringIO(f.read().decode('utf-8', errors='replace')), sep=',', engine='python', on_bad_lines='skip', header=0)
            dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print(f"Full train dataset: {df.shape}")

# Step 2: Preprocess (same as before)
feature_cols = ['Time', 'ID', 'Signal1', 'Signal2', 'Signal3', 'Signal4']
X = df[feature_cols].copy()

signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']
imputer = SimpleImputer(strategy='constant', fill_value=0)
X[signal_cols] = imputer.fit_transform(X[signal_cols])

X_sorted = X.sort_values(['ID', 'Time']).reset_index(drop=True)

X_sorted['Time_delta'] = X_sorted.groupby('ID')['Time'].diff().fillna(1.0)

for sig in signal_cols:
    X_sorted[f'{sig}_delta'] = X_sorted.groupby('ID')[sig].diff().fillna(0)
    X_sorted[f'{sig}_abs_delta'] = np.abs(X_sorted[f'{sig}_delta'])

for sig in signal_cols:
    delta_col = f'{sig}_delta'
    X_sorted[f'{delta_col}_roll_var'] = X_sorted.groupby('ID')[delta_col].transform(lambda x: x.rolling(5, min_periods=1).var()).fillna(0)
    X_sorted[f'{delta_col}_roll_mean'] = X_sorted.groupby('ID')[delta_col].transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0)

X = X_sorted.drop(['Time'] + signal_cols, axis=1).sort_index().reset_index(drop=True)

le = LabelEncoder()
X['ID_encoded'] = le.fit_transform(X['ID'])
X = X.drop('ID', axis=1)

numeric_cols = [col for col in X.columns if col != 'ID_encoded']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"Preprocessed X shape: {X.shape}")

X = X.sample(frac=0.05, random_state=42)  # Subsample
print(f"Subsampled X shape: {X.shape}")

# Step 3: Train Ensemble
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# IF
if_model = IsolationForest(n_estimators=200, contamination=0.001, random_state=42, n_jobs=-1)
if_model.fit(X_train)

# OCSVM
ocsvm_model = OneClassSVM(kernel='rbf', nu=0.001, gamma='scale')
ocsvm_model.fit(X_train)

# Save
joblib.dump({'if': if_model, 'ocsvm': ocsvm_model}, 'syncan_ensemble_model.pkl')
joblib.dump(le, 'id_encoder.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Ensemble artifacts saved!")

# Val check (average inverted scores - lower = anomaly)
if_scores = if_model.decision_function(X_val)
ocsvm_scores = ocsvm_model.score_samples(X_val)
ensemble_scores = (if_scores + ocsvm_scores) / 2  # Average (higher = normal)
ensemble_pred = (ensemble_scores < np.percentile(ensemble_scores, 99))  # Top 1% anomaly (invert percentile)
fraction_anomalies = ensemble_pred.mean()
print(f"Fraction predicted anomalies on val: {fraction_anomalies:.4f}")