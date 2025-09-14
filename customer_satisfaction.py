# customer_satisfaction.py

# ===============================
# Step 1: Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Avoid Tkinter freezing on Windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ===============================
# Step 2: Load Dataset
# ===============================
print("Loading dataset...")
data = pd.read_csv("customer_support_tickets.csv")

print("\nFirst 5 rows:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# ===============================
# Step 3: Data Preprocessing
# ===============================
# Drop rows with missing satisfaction rating
data = data.dropna(subset=["Customer Satisfaction Rating"])

# Convert Date of Purchase to datetime
data["Date of Purchase"] = pd.to_datetime(data["Date of Purchase"], errors="coerce")
data["Purchase_Year"] = data["Date of Purchase"].dt.year
data["Purchase_Month"] = data["Date of Purchase"].dt.month

# Age groups
bins = [0, 20, 30, 40, 50, 60, 70, 100]
labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71+"]
data["Age_Group"] = pd.cut(data["Customer Age"], bins=bins, labels=labels, right=False)

# Encode categorical variables (including Age_Group)
for col in data.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

print("\nColumns after encoding:")
print(data.columns)

# ===============================
# Step 4: Feature & Target Split
# ===============================
X = data.drop([
    'Customer Satisfaction Rating',
    'Ticket ID',
    'Customer Name',
    'Customer Email',
    'Date of Purchase'  # drop raw date
], axis=1, errors='ignore')

y = data['Customer Satisfaction Rating']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# ===============================
# Step 5: Multiple Model Comparison (Multi-class 1–5)
# ===============================
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

print("\n🔹 Multi-class Classification (1–5 Ratings)")
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

# ===============================
# Step 6: Binary Classification (Satisfied vs Not)
# ===============================
# Define: Satisfied (4,5) = 1 | Not satisfied (1–3) = 0
y_binary = y.apply(lambda x: 1 if x >= 4 else 0)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_scaled, y_binary, test_size=0.3, random_state=42
)

print("\n🔹 Binary Classification (Satisfied vs Not Satisfied)")
for name, clf in models.items():
    clf.fit(X_train_b, y_train_b)
    y_pred_b = clf.predict(X_test_b)
    acc_b = accuracy_score(y_test_b, y_pred_b)
    print(f"\n{name} Accuracy: {acc_b:.2f}")
    print(classification_report(y_test_b, y_pred_b))

# ===============================
# Step 7: GridSearchCV for Random Forest (Binary Task)
# ===============================
print("\n🔹 Running GridSearchCV for Random Forest (Binary Classification)...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=3, scoring='accuracy', n_jobs=-1)

grid.fit(X_train_b, y_train_b)

print("Best Params:", grid.best_params_)
print("Best Cross-validation Accuracy:", grid.best_score_)

best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test_b)
print("Test Accuracy with Best RF:", accuracy_score(y_test_b, y_pred_best))

import joblib
# Save the trained best Random Forest model
joblib.dump(best_rf, "best_rf_model.pkl")
print("✅ Model saved as best_rf_model.pkl")


# ===============================
# Step 8: Feature Importance (Save as Image)
# ===============================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 10 Feature Importances (Multi-class RF)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig("feature_importances.png")
print("✅ Feature importance plot saved as feature_importances.png")
