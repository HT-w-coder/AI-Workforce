# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("Employee.csv")

# Drop nulls if any
df.dropna(inplace=True)

# Features and target
X = df.drop(columns=["LeaveOrNot"])
y = df["LeaveOrNot"]

# Separate categorical and numerical
categorical_cols = ["Education", "City", "Gender", "EverBenched"]
numerical_cols = ["JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain"]

# OneHot encode categorical columns
encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

# Combine with numerical data
X_full = np.hstack((X[numerical_cols].values, X_encoded.toarray()))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model, encoder, accuracy
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(accuracy, "accuracy.pkl")

print(f"Training complete. Accuracy: {accuracy:.4f}")
