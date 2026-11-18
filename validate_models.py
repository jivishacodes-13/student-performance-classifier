from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")


# 2. Create target variable
def classify_performance(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Average"
    else:
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


# 4. Keep the final test set untouched
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 5. Preprocessing function
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
    "Dummy Baseline": (
        DummyClassifier(strategy="most_frequent"),
        False
    ),

    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        True
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        False
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ),
        False
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
        False
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

print("=" * 65)
print("5-FOLD CROSS-VALIDATION")
print("=" * 65)

print("\nTraining records:", len(X_train))
print("Reserved test records:", len(X_test))


for name, (model, scale_numeric) in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(scale_numeric)),
            ("model", model)
        ]
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "Mean Accuracy": scores["test_accuracy"].mean(),
        "Accuracy Std": scores["test_accuracy"].std(),
        "Mean Balanced Accuracy": (
            scores["test_balanced_accuracy"].mean()
        ),
        "Mean Macro F1": scores["test_macro_f1"].mean(),
        "Macro F1 Std": scores["test_macro_f1"].std()
    })


# 8. Display results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Mean Macro F1",
    ascending=False
)

print("\nCROSS-VALIDATION RESULTS")
print("-" * 65)

print(results_df.round(4).to_string(index=False))


# 9. Save results
output_path = BASE_DIR / "cross_validation_results.csv"

results_df.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
print("\nCross-validation completed!")