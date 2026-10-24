import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_performance_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Performance Classifier",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 Student Performance Classifier")

st.markdown(
    """
    ### Understand academic performance through machine learning

    Explore how study habits, previous academic challenges, and
    student-related factors relate to predicted performance.

    This application uses a machine learning model to estimate
    one of three categories: **Low, Average, or High**.
    """
)

st.info(
    "This is an experimental prediction tool, not a definitive "
    "assessment of a student's abilities. Predictions may be "
    "incorrect and should not replace individual academic guidance."
)

st.divider()
# Input section
st.header("📋 Student Information")

col1, col2 = st.columns(2)

with col1:
    studytime = st.slider(
        "Weekly Study Time",
        1, 4, 2,
        help="1: <2 hours | 2: 2-5 hours | 3: 5-10 hours | 4: >10 hours"
    )

    failures = st.number_input(
        "Past Class Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=93,
        value=5
    )

    health = st.slider(
        "Current Health",
        1, 5, 3
    )

with col2:
    Medu = st.slider(
        "Mother's Education Level",
        0, 4, 2
    )

    Fedu = st.slider(
        "Father's Education Level",
        0, 4, 2
    )

    schoolsup = st.selectbox(
        "Extra School Support",
        ["yes", "no"]
    )

    famsup = st.selectbox(
        "Family Educational Support",
        ["yes", "no"]
    )

higher = st.selectbox(
    "Wants Higher Education",
    ["yes", "no"]
)

internet = st.selectbox(
    "Internet Access at Home",
    ["yes", "no"]
)

st.divider()

# Prediction button
if st.button("🔮 Predict Performance", use_container_width=True):

    input_data = pd.DataFrame([{
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "Medu": Medu,
        "Fedu": Fedu,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "higher": higher,
        "internet": internet,
        "health": health
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(input_data)[0]
    class_names = model.classes_

    confidence = probabilities[
        list(class_names).index(prediction)
    ]

    # -------------------------------
    # PREDICTION RESULT
    # -------------------------------

    st.subheader("📊 Prediction Result")

    if prediction == "High":
        st.success("### 🟢 HIGH PERFORMANCE")

    elif prediction == "Average":
        st.warning("### 🟡 AVERAGE PERFORMANCE")

    else:
        st.error("### 🔴 LOW PERFORMANCE")

    st.metric(
    "Estimated Model Confidence",
    f"{confidence * 100:.2f}%"
    )

    st.caption(
        "This score represents the model's estimated probability "
        "for the predicted category. It is not a guarantee of "
        "correctness, and the probabilities have not necessarily "
        "been calibrated."
    )

    st.info(
        "This tool provides an experimental academic performance "
        "estimate, not a definitive assessment of a student. "
        "Use the result alongside other relevant information."
    )

    # -------------------------------
    # ACADEMIC RISK
    # -------------------------------

    st.subheader("⚠️ Academic Risk")

    if prediction == "Low":
        st.error(
            "High Risk: This student may need additional academic support."
        )

    elif prediction == "Average":
        st.warning(
            "Moderate Risk: There is room for academic improvement."
        )

    else:
        st.success(
            "Low Risk: The student is currently predicted to perform well."
        )

    # -------------------------------
    # TOP CONTRIBUTING FACTORS
    # -------------------------------

    st.subheader("🔍 Top Factors Influencing Prediction")

    try:
        preprocessor = model.named_steps["preprocessor"]
        rf_model = model.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()
        importances = rf_model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        # Convert encoded feature names back to original feature names
        def clean_feature_name(name):
            name = name.replace("cat__", "")
            name = name.replace("num__", "")

            if name.startswith("schoolsup_"):
                return "Extra School Support"
            elif name.startswith("famsup_"):
                return "Family Educational Support"
            elif name.startswith("higher_"):
                return "Higher Education Goal"
            elif name.startswith("internet_"):
                return "Internet Access"
            elif name == "studytime":
                return "Study Time"
            elif name == "failures":
                return "Past Failures"
            elif name == "absences":
                return "Absences"
            elif name == "Medu":
                return "Mother's Education"
            elif name == "Fedu":
                return "Father's Education"
            elif name == "health":
                return "Health"
            else:
                return name

        importance_df["Feature"] = importance_df["Feature"].apply(
            clean_feature_name
        )

        # Combine one-hot encoded features belonging to the same factor
        importance_df = (
            importance_df
            .groupby("Feature", as_index=False)["Importance"]
            .sum()
            .sort_values("Importance", ascending=False)
            .head(5)
        )

        for _, row in importance_df.iterrows():
            st.write(
                f"**{row['Feature']}** — "
                f"{row['Importance'] * 100:.2f}%"
            )

            st.progress(float(row["Importance"]))

    except Exception:
        st.info(
            "Feature importance could not be displayed for this model."
        )

    # -------------------------------
    # IMPROVEMENT SUGGESTIONS
    # -------------------------------

    st.subheader("💡 Improvement Suggestions")

    suggestions = []

    if studytime <= 2:
        suggestions.append(
            "Increase focused weekly study time."
        )

    if failures > 0:
        suggestions.append(
            "Focus on subjects where previous failures occurred."
        )

    if absences > 10:
        suggestions.append(
            "Improve class attendance and reduce unnecessary absences."
        )

    if schoolsup == "no":
        suggestions.append(
            "Consider using additional academic support when needed."
        )

    if famsup == "no":
        suggestions.append(
            "Seek support from family or mentors when facing academic difficulties."
        )

    if prediction == "Low" and not suggestions:
        suggestions.append(
            "Increase study consistency and consider additional academic support."
        )

    elif prediction == "Average" and not suggestions:
        suggestions.append(
            "Increase study consistency to improve academic performance."
        )

    elif prediction == "High" and not suggestions:
        suggestions.append(
            "Maintain your current study habits and academic routine."
        )

    for suggestion in suggestions:
        st.write("•", suggestion)