from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")


# 2. Create target categories
def classify_performance(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Average"
    return "Low"


df["Performance"] = df["G3"].apply(classify_performance)


# 3. Define features
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


# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 5. Define preprocessing
def create_preprocessor(scale_numeric=False):
    numeric_transformer = (
        StandardScaler() if scale_numeric else "passthrough"
    )

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


# 6. Define models
models = {
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ),
        False
    ),

    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        True
    )
}


# 7. Cross-validation on training data only
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro"
}

results = []

print("=" * 75)
print("FINAL MODEL COMPARISON — 5-FOLD CROSS-VALIDATION")
print("=" * 75)

print("Training samples:", len(X_train))
print("Reserved test samples:", len(X_test))

for name, (estimator, scale_numeric) in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(scale_numeric)),
            ("model", estimator)
        ]
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "Mean Accuracy": scores["test_accuracy"].mean(),
        "Mean Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
        "Mean Macro F1": scores["test_macro_f1"].mean(),
        "Macro F1 Std": scores["test_macro_f1"].std()
    })


# 8. Display and save results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Mean Macro F1",
    ascending=False
)

print("\nCROSS-VALIDATION RESULTS")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    BASE_DIR / "final_model_comparison.csv",
    index=False
)

print("\nResults saved to final_model_comparison.csv")
print("\nComparison completed!")