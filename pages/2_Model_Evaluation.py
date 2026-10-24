from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATASET AND MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "student-mat.csv",
    sep=";"
)

model = joblib.load(
    BASE_DIR / "student_performance_model.pkl"
)


# ==========================================
# 2. PREPARE TARGET
# ==========================================

def classify_performance(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Average"
    return "Low"


df["Performance"] = df["G3"].apply(
    classify_performance
)


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

X = df[features]
y = df["Performance"]


# ==========================================
# 4. REPRODUCE TEST SPLIT
# ==========================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)


# ==========================================
# 5. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


# ==========================================
# 6. STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Student Performance Model Evaluation")

st.write(
    "Evaluate the performance of the trained "
    "Student Performance Classifier."
)

st.caption(
    "These results use a held-out split from the development "
    "dataset. Repeated experimentation on this split means "
    "they should not be treated as a fully independent final evaluation."
)


# ==========================================
# 7. METRIC CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Balanced Accuracy",
    f"{balanced_accuracy * 100:.2f}%"
)

col3.metric(
    "Macro F1-score",
    f"{macro_f1 * 100:.2f}%"
)


st.divider()


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

st.subheader("Confusion Matrix")

labels = ["Low", "Average", "High"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax
)

ax.set_xlabel("Predicted Category")
ax.set_ylabel("Actual Category")
ax.set_title("Actual vs Predicted Performance")

fig.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

st.subheader("Class-wise Performance")

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)


# ==========================================
# 10. LIMITATIONS
# ==========================================

st.subheader("Model Limitations")

st.warning(
    """
    - The model has limited predictive performance on this dataset.
    - Performance differs across the three categories.
    - Results may vary with different training and test samples.
    - Predictions should not be used as definitive judgments
      about individual students.
    - Further validation is needed before any real-world use.
    """
)

st.success("Model evaluation completed.")