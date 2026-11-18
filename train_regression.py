from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer
)


# 1. Load dataset
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")


# 2. Select features
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
y = df["G3"]


# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 4. Preprocessing
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


# 5. Define models
models = {
    "Dummy Baseline": (
        DummyRegressor(strategy="mean"),
        False
    ),

    "Ridge Regression": (
        Ridge(alpha=1.0),
        True
    ),

    "Random Forest Regressor": (
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=3
        ),
        False
    )
}


# 6. Cross-validation
cv_results = []

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
        cv=5,
        scoring={
            "mae": "neg_mean_absolute_error",
            "r2": "r2"
        },
        n_jobs=-1
    )

    cv_results.append({
        "Model": name,
        "Mean CV MAE": -scores["test_mae"].mean(),
        "Mean CV R2": scores["test_r2"].mean()
    })


# 7. Display cross-validation results
results_df = pd.DataFrame(cv_results)

print("\nCROSS-VALIDATION RESULTS")
print(results_df.round(3).to_string(index=False))


# 8. Evaluate each model on the same reserved test set
print("\nFINAL TEST-SET RESULTS")

for name, (model, scale_numeric) in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(scale_numeric)),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2 Score: {r2:.3f}")


print("\nRegression experiment completed!")