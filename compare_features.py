from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


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

y = df["Performance"]


# 3. Define feature sets
all_features = [
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

selected_features = [
    "failures",
    "health",
    "Medu",
    "absences",
    "studytime"
]


# 4. Reserve test data
X_train_all, X_test_all, y_train, y_test = train_test_split(
    df[all_features],
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train_selected = X_train_all[selected_features]


# 5. Feature groups
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


# 6. Model evaluation function
def evaluate_features(X_train, feature_list, name):

    cat_features = [
        feature for feature in categorical_features
        if feature in feature_list
    ]

    num_features = [
        feature for feature in numeric_features
        if feature in feature_list
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_features
            ),
            (
                "num",
                "passthrough",
                num_features
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro"
        },
        n_jobs=-1
    )

    return {
        "Feature Set": name,
        "Mean Accuracy": scores["test_accuracy"].mean(),
        "Mean Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
        "Mean Macro F1": scores["test_macro_f1"].mean(),
        "Macro F1 Std": scores["test_macro_f1"].std()
    }


# 7. Compare both feature sets
results = []

results.append(
    evaluate_features(
        X_train_all,
        all_features,
        "All 10 Features"
    )
)

results.append(
    evaluate_features(
        X_train_selected,
        selected_features,
        "Selected 5 Features"
    )
)


# 8. Display results
results_df = pd.DataFrame(results)

print("\nFEATURE SET COMPARISON")
print("=" * 65)

print(results_df.round(4).to_string(index=False))


# 9. Save results
results_df.to_csv(
    BASE_DIR / "feature_set_comparison.csv",
    index=False
)

print("\nResults saved to feature_set_comparison.csv")
print("\nComparison completed!")