from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)


# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent

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


# 4. Reserve test data
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


# 6. Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                random_state=42
            )
        )
    ]
)


# 7. Hyperparameter search
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],

    "model__class_weight": [
        None,
        "balanced"
    ],

    "model__solver": [
        "lbfgs"
    ]
}


# 8. Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True
)


print("Starting Logistic Regression tuning...")

search.fit(X_train, y_train)


# 9. Show best parameters
print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation Macro F1:")
print(round(search.best_score_, 4))


# 10. Final evaluation on reserved test data
best_model = search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nFINAL TEST RESULTS")

print(
    "Accuracy:",
    round(accuracy_score(y_test, y_pred), 4)
)

print(
    "Balanced Accuracy:",
    round(balanced_accuracy_score(y_test, y_pred), 4)
)

print(
    "Macro F1:",
    round(f1_score(y_test, y_pred, average="macro"), 4)
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Low", "Average", "High"],
        zero_division=0
    )
)


# 11. Save tuning results
pd.DataFrame(search.cv_results_).to_csv(
    BASE_DIR / "logistic_tuning_results.csv",
    index=False
)

print("\nResults saved to logistic_tuning_results.csv")
print("Tuning completed!")