from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# ==========================================
# 1. LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")


# ==========================================
# 2. CREATE PERFORMANCE CATEGORIES
# ==========================================

def classify_performance(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Average"
    return "Low"


df["Performance"] = df["G3"].apply(classify_performance)


# ==========================================
# 3. SELECT EARLY-PREDICTION FEATURES
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
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. BUILD PIPELINE
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


# ==========================================
# 6. TRAIN MODEL
# ==========================================

pipeline.fit(X_train, y_train)


# ==========================================
# 7. CALCULATE PERMUTATION IMPORTANCE
# ==========================================

importance = permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="f1_macro",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance.importances_mean,
    "Std": importance.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("=" * 50)

print(importance_df.round(4).to_string(index=False))


# ==========================================
# 8. SAVE RESULTS
# ==========================================

output_dir = BASE_DIR / "visualizations"
output_dir.mkdir(exist_ok=True)

importance_df.to_csv(
    BASE_DIR / "feature_importance.csv",
    index=False
)


# ==========================================
# 9. VISUALIZE RESULTS
# ==========================================

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(9, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    ax=ax
)

ax.axvline(0, linestyle="--", linewidth=1)

ax.set_title("Permutation Feature Importance")
ax.set_xlabel("Change in Macro F1 after Permutation")
ax.set_ylabel("Feature")

fig.tight_layout()

fig.savefig(
    output_dir / "feature_importance.png",
    dpi=300
)

plt.close(fig)

print("\nFeature importance analysis completed!")
print("Results saved to feature_importance.csv")
print("Graph saved to visualizations/feature_importance.png")