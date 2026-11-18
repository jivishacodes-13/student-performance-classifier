from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")

print("Dataset loaded:", df.shape)


# ==========================================
# 2. CREATE TARGET
# ==========================================

def classify_performance(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Average"
    return "Low"


df["Performance"] = df["G3"].apply(classify_performance)


# ==========================================
# 3. SELECT FEATURES
# ==========================================

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


# ==========================================
# 4. RESERVE TEST SET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Reserved test records:", len(X_test))


# ==========================================
# 5. PREPROCESSING PIPELINE
# ==========================================

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


# ==========================================
# 6. RANDOM FOREST PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(random_state=42)
        )
    ]
)


# ==========================================
# 7. DEFINE HYPERPARAMETERS
# ==========================================

param_grid = {
    "model__n_estimators": [100, 200, 300],

    "model__max_depth": [None, 5, 10],

    "model__min_samples_split": [2, 5, 10],

    "model__min_samples_leaf": [1, 2, 4],

    "model__class_weight": ["balanced", None]
}


# ==========================================
# 8. CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 9. GRID SEARCH
# ==========================================

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=False
)

print("\nStarting hyperparameter tuning...")
print("This may take a few minutes.")

search.fit(X_train, y_train)


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 55)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 55)

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest Cross-Validation Macro F1:")
print(round(search.best_score_, 4))


# ==========================================
# 11. SAVE SEARCH RESULTS
# ==========================================

results_df = pd.DataFrame(search.cv_results_)

results_df = results_df.sort_values(
    by="mean_test_score",
    ascending=False
)

results_df.to_csv(
    BASE_DIR / "random_forest_tuning_results.csv",
    index=False
)

print("\nTuning results saved to:")
print(BASE_DIR / "random_forest_tuning_results.csv")


# ==========================================
# 12. EVALUATE BEST MODEL ON TEST SET
# ==========================================

best_model = search.best_estimator_

y_pred = best_model.predict(X_test)

print("\n" + "=" * 55)
print("FINAL TEST SET EVALUATION")
print("=" * 55)

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

print("\nTuning and evaluation completed!")