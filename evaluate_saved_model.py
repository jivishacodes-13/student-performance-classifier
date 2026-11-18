from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent

# Load dataset and saved model
df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")
model = joblib.load(BASE_DIR / "student_performance_model.pkl")


# Create target
def classify_performance(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Average"
    return "Low"


df["Performance"] = df["G3"].apply(classify_performance)


# Use the same features as the application
features = [
    "studytime",
    "failures",
    "absences",
    "Medu",
    "Fedu",
    "schoolsup",
    "famsup",
    "higher",
    "internet",
    "health"
]

X = df[features]
y = df["Performance"]


# Reproduce the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Evaluate the saved model
predictions = model.predict(X_test)

print("=" * 55)
print("SAVED MODEL EVALUATION")
print("=" * 55)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")

print(
    "Balanced Accuracy: "
    f"{balanced_accuracy_score(y_test, predictions):.4f}"
)

print(
    "Macro F1-score: "
    f"{f1_score(y_test, predictions, average='macro'):.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=["Low", "Average", "High"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["Low", "Average", "High"]
    )
)

print("\nEvaluation completed.")