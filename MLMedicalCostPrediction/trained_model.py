import pandas as pd 
import numpy as np
import joblib as jl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/preparedData.csv")
df = df.fillna(df.mean(numeric_only=True))

#Target value to be predicted, charges
y = df['charges']
#Selected features for the prediction 
X = df.drop('charges', axis=1)

#Split in test and train 20-80
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Converting categories in numerial features that the model can understand
#One hot encode 
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

#Align via reindex
X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)

#Scale feature, standartization
numeric_features = ["age", "bmi", "children"]

scaler = StandardScaler()

X_train_scaled = X_train_encoded.copy() 
X_test_scaled = X_test_encoded.copy()

# fit on training data to ensure that scaling params are learned strictly from the train set
X_train_scaled[numeric_features] = scaler.fit_transform(
    X_train_encoded[numeric_features]
)

# use same scaler to transform the test, prevents data leakage
X_test_scaled[numeric_features] = scaler.transform(
    X_test_encoded[numeric_features]
)

#Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

coeff = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coef': model.coef_
})

print(coeff)

#Baseline value
intercept = model.intercept_
print(intercept)

jl.dump(model, "model.pkl")
jl.dump(scaler, "scaler.pkl")