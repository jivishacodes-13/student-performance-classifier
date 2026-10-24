from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


# ==================================================
# 1. LOAD DATASET
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")

print("=" * 60)
print("STUDENT PERFORMANCE CLASSIFIER")
print("=" * 60)

print("\nDataset shape:", df.shape)


# ==================================================
# 2. CREATE TARGET VARIABLE
# ==================================================

def classify_performance(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Average"
    else:
        return "Low"


df["Performance"] = df["G3"].apply(classify_performance)

print("\nTarget distribution:")
print(df["Performance"].value_counts())


# ==================================================
# 3. SELECT FEATURES
# ==================================================

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


# ==================================================
# 4. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==================================================
# 5. PREPROCESSING
# ==================================================

def create_preprocessor(scale_numeric=False):

    if scale_numeric:
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = "passthrough"

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "num",
                numeric_transformer,
                numeric_features
            )
        ]
    )


# ==================================================
# 6. DEFINE MODELS
# ==================================================

models = {
    "Dummy Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==================================================
# 7. TRAIN AND EVALUATE MODELS
# ==================================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining: {name}")

    scale_numeric = name == "Logistic Regression"

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(scale_numeric)),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1
    })

    trained_models[name] = pipeline

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


# ==================================================
# 8. COMPARE MODELS
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

results_df.to_csv(
    BASE_DIR / "model_comparison.csv",
    index=False
)


# ==================================================
# 9. SELECT BEST MODEL
# ==================================================

best_model_name = results_df.iloc[0]["Model"]

best_pipeline = trained_models[best_model_name]

print("\nBest model by test-set Macro F1:", best_model_name)


# ==================================================
# 10. FINAL EVALUATION
# ==================================================

y_pred = best_pipeline.predict(X_test)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["Low", "Average", "High"],
        zero_division=0
    )
)


# ==================================================
# 11. SAVE CONFUSION MATRIX
# ==================================================

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["Low", "Average", "High"],
    display_labels=["Low", "Average", "High"],
    cmap="Blues",
    ax=ax
)

ax.set_title(f"Confusion Matrix - {best_model_name}")

fig.tight_layout()

fig.savefig(
    BASE_DIR / "visualizations" / "confusion_matrix.png",
    dpi=300
)

plt.close(fig)


# ==================================================
# 12. SAVE MODEL
# ==================================================

joblib.dump(
    best_pipeline,
    BASE_DIR / "student_performance_model.pkl"
)

print("\nBest model saved successfully.")

print("Model file: student_performance_model.pkl")
print("Comparison file: model_comparison.csv")
print("Confusion matrix saved in visualizations/")

print("\nTraining and evaluation completed!")