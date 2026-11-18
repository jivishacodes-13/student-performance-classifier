from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score
)


# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")


# 2. Create target variable
def classify_performance(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Average"
    return "Low"


df["Performance"] = df["G3"].apply(classify_performance)


# 3. Select features
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

categorical_features = [
    "schoolsup",
    "famsup",
    "higher",
    "internet"
]

numeric_features = [
    "studytime",
    "failures",
    "absences",
    "Medu",
    "Fedu",
    "health"
]

X = df[features]
y = df["Performance"]


# 4. Reproduce the same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            StandardScaler(),
            numeric_features
        )
    ]
)


# 6. Create Logistic Regression pipeline
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# 7. Train
logistic_pipeline.fit(X_train, y_train)


# 8. Predict
y_pred = logistic_pipeline.predict(X_test)


# 9. Evaluate
print("=" * 50)
print("LOGISTIC REGRESSION - TEST EVALUATION")
print("=" * 50)

print(
    f"Accuracy: {accuracy_score(y_test, y_pred):.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy_score(y_test, y_pred):.4f}"
)

print(
    f"Macro F1-score: "
    f"{f1_score(y_test, y_pred, average='macro'):.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Low", "Average", "High"],
        zero_division=0
    )
)


# 10. Compare against the baseline
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

print("\nBASELINE COMPARISON")
print("-" * 50)

print(
    f"Baseline Accuracy: "
    f"{accuracy_score(y_test, baseline_pred):.4f}"
)

print(
    f"Baseline Macro F1: "
    f"{f1_score(y_test, baseline_pred, average='macro'):.4f}"
)


print("\nEvaluation completed!")