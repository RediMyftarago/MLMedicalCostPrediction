import numpy as np 
import joblib as jl
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/preparedData.csv")
df = df.fillna(df.mean(numeric_only=True))

X = df.drop('charges', axis=1)
y = df['charges']

# Load model and scaler
model = jl.load("models/model.pkl")
scaler = jl.load("models/scaler.pkl")

X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded = X_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Scale
numeric_features = ["age", "bmi", "children"]
X_encoded[numeric_features] = scaler.transform(X_encoded[numeric_features])

# Predict
y_pred = model.predict(X_encoded)

# Evaluate
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)