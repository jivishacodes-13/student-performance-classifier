from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent

# 1. Load dataset
df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")


# 2. Create target
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
            "passthrough",
            numeric_features
        )
    ]
)


# 6. Define the candidate model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 7. Train on training data only
pipeline.fit(X_train, y_train)


# 8. Evaluate once on the reserved test set
y_pred = pipeline.predict(X_test)

print("=" * 55)
print("FINAL TEST-SET EVALUATION")
print("=" * 55)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

print(
    "Balanced Accuracy:",
    f"{balanced_accuracy_score(y_test, y_pred):.4f}"
)

print(
    "Macro F1:",
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


# 9. Save confusion matrix
output_dir = BASE_DIR / "visualizations"
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["Low", "Average", "High"],
    display_labels=["Low", "Average", "High"],
    cmap="Blues",
    ax=ax
)

ax.set_title("Random Forest - Final Test Results")
fig.tight_layout()

fig.savefig(
    output_dir / "final_confusion_matrix.png",
    dpi=300
)

plt.close(fig)


# 10. Save the fitted pipeline
import joblib

joblib.dump(
    pipeline,
    BASE_DIR / "student_performance_model.pkl"
)

print("\nConfusion matrix saved.")
print("Model saved successfully.")
print("\nEvaluation completed!")