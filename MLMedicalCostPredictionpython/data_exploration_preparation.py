import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/insurance.csv")

df.head()
df.info()

df.isna().sum() #no NaN values

df.duplicated().sum() #1 duplicate 
df.drop_duplicates(inplace=True) #remove the duplicate, can cause overfit in the model

#Map gender column from female and male to 0 and 1
df['sex'] = df['sex'].map({'female': 0, 'male': 1})

#Map children columns from yes and no to 1 and 0
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1 })

#Unique values of region column
print(df['region'].unique()) #4

#Target prediction charges
print(df['charges'].corr(df['bmi'])) #0.198
print(df['charges'].corr(df['age'])) #0.298
print(df['charges'].corr(df['sex'])) #0.058
print(df['charges'].corr(df['children'])) #0.067
print(df['charges'].corr(df['smoker'])) #0.787, strong positive correlation

# One-hot encoding for the region column
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['region']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['region']))
df = pd.concat([df, encoded_df], axis=1)
df.drop('region', axis=1, inplace=True)

#Save the prepared data in the data folder
df.to_csv("data/preparedData.csv", index=False)


