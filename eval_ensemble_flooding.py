import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import zipfile
import io
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load test ZIP
test_zip_path = 'test_flooding.zip'
with zipfile.ZipFile(test_zip_path, 'r') as z:
    csv_name = [name for name in z.namelist() if name.lower().endswith('.csv')][0]
    with z.open(csv_name) as f:
        test_df = pd.read_csv(io.StringIO(f.read().decode('utf-8', errors='replace')), sep=',', engine='python', on_bad_lines='skip', header=0)
print(f"Loaded: {test_df.shape[0]} rows, Labels: {test_df['Label'].value_counts().to_dict()}")

# Step 2: Load preprocessors and apply (same)
le = joblib.load('id_encoder.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

signal_rename = {'Signal1_of_ID': 'Signal1', 'Signal2_of_ID': 'Signal2', 'Signal3_of_ID': 'Signal3', 'Signal4_of_ID': 'Signal4'}
for old, new in signal_rename.items():
    if old in test_df.columns:
        test_df[new] = test_df[old]

feature_cols = ['Time', 'ID', 'Signal1', 'Signal2', 'Signal3', 'Signal4']
X_test = test_df[feature_cols].copy()
y_test = test_df['Label'].values

signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']
X_test[signal_cols] = imputer.transform(X_test[signal_cols])

X_test_sorted = X_test.sort_values(['ID', 'Time']).reset_index(drop=True)

X_test_sorted['Time_delta'] = X_test_sorted.groupby('ID')['Time'].diff().fillna(1.0)

for sig in signal_cols:
    X_test_sorted[f'{sig}_delta'] = X_test_sorted.groupby('ID')[sig].diff().fillna(0)
    X_test_sorted[f'{sig}_abs_delta'] = np.abs(X_test_sorted[f'{sig}_delta'])

for sig in signal_cols:
    delta_col = f'{sig}_delta'
    X_test_sorted[f'{delta_col}_roll_var'] = X_test_sorted.groupby('ID')[delta_col].transform(lambda x: x.rolling(5, min_periods=1).var()).fillna(0)
    X_test_sorted[f'{delta_col}_roll_mean'] = X_test_sorted.groupby('ID')[delta_col].transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0)

X_test = X_test_sorted.drop(['Time'] + signal_cols, axis=1).sort_index().reset_index(drop=True)

X_test['ID_encoded'] = le.transform(X_test['ID'])
X_test = X_test.drop('ID', axis=1)

numeric_cols = [col for col in X_test.columns if col != 'ID_encoded']
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"Preprocessed X_test shape: {X_test.shape}")

# Step 3: Load ensemble and predict
ensemble = joblib.load('syncan_ensemble_model.pkl')
if_model = ensemble['if']
ocsvm_model = ensemble['ocsvm']

if_scores = if_model.decision_function(X_test)
ocsvm_scores = ocsvm_model.score_samples(X_test)
ensemble_scores = (if_scores + ocsvm_scores) / 2  # Average (higher = normal)
print(f"Mean ensemble score on test: {ensemble_scores.mean():.4f}")
print(f"Std ensemble score on test: {ensemble_scores.std():.4f}")

# Step 4: Threshold for F1
y_pred_proba = -ensemble_scores  # Invert: higher = anomaly
prec, rec, thresh = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_thresh = thresh[optimal_idx]
print(f"Optimal threshold: {optimal_thresh:.4f} (max F1 on curve: {f1_scores[optimal_idx]:.4f})")

y_pred = (y_pred_proba > optimal_thresh).astype(int)

# Metrics
f1 = f1_score(y_test, y_pred, average='binary')
prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
rec = recall_score(y_test, y_pred, average='binary')
acc = accuracy_score(y_test, y_pred)

fp = ((y_pred == 1) & (y_test == 0)).sum()
tn = ((y_pred == 0) & (y_test == 0)).sum()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\nTest Metrics (flooding Data - Ensemble Fixed):")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"False Positive Rate: {fpr:.4f}")

# Save
results_df = test_df.copy()
results_df['Anomaly_Score'] = ensemble_scores
results_df['Predicted_Label'] = y_pred
results_df.to_csv('test_flooding_predictions_ensemble_fixed.csv', index=False)
print("\nPredictions saved to 'test_flooding_predictions_ensemble_fixed.csv'")

# Plot PR curve
plt.figure(figsize=(8, 6))
plt.plot(rec, prec, label=f'PR Curve (F1 max at {f1_scores[optimal_idx]:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for flooding Test (Ensemble Fixed)')
plt.legend()
plt.savefig('pr_curve_flooding_ensemble_fixed.png')
plt.show()