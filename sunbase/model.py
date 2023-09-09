import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


data=pd.read_csv('customer_churn_large_dataset.csv')
data.head()

data.shape
data.isnull().sum()
data.value_counts('Gender')
data.value_counts('Location')
data = pd.get_dummies(data, columns=['Gender', 'Location'])
x=data.iloc[:,2:-1]
y=data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}")

import pickle
pickle.dump(model, open('model.pkl','wb'))
