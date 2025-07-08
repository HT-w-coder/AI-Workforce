# retrain_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
df = pd.read_csv("Employee.csv")

# Encode categorical columns
label_encoders = {}
for col in ['Education', 'City', 'Gender', 'EverBenched']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop("LeaveOrNot", axis=1)
y = df["LeaveOrNot"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "encoder.pkl")
joblib.dump(accuracy, "accuracy.pkl")
