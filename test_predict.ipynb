import joblib
import pandas as pd

model_path = r"C:\Users\admin\Downloads\archive (3)\diabetes_random_forest_model.pkl"
features_path = r"C:\Users\admin\Downloads\archive (3)\selected_features.txt"

model = joblib.load(model_path)
with open(features_path, 'r') as f:
    features = [l.strip() for l in f if l.strip()]

# Build a sample input aligned to the features
sample = {feat: 0 for feat in features}
sample['age'] = 30
if 'hba1c' in sample:
    sample['hba1c'] = 5.5
if 'glucose_fasting' in sample:
    sample['glucose_fasting'] = 100
if 'glucose_postprandial' in sample:
    sample['glucose_postprandial'] = 100

# Create DataFrame in expected order
X = pd.DataFrame([sample], columns=features)
print('Input shape:', X.shape)
print('Columns (first 20):', X.columns.tolist()[:20])

pred = model.predict(X)
print('Prediction:', pred)
