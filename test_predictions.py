from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "student_performance_model.pkl"
)

# Test different input combinations
test_cases = [
    {
        "studytime": 4,
        "failures": 0,
        "absences": 0,
        "Medu": 4,
        "Fedu": 4,
        "schoolsup": "yes",
        "famsup": "yes",
        "higher": "yes",
        "internet": "yes",
        "health": 5,
    },
    {
        "studytime": 4,
        "failures": 0,
        "absences": 0,
        "Medu": 2,
        "Fedu": 2,
        "schoolsup": "no",
        "famsup": "no",
        "higher": "yes",
        "internet": "yes",
        "health": 5,
    },
    {
        "studytime": 3,
        "failures": 0,
        "absences": 2,
        "Medu": 3,
        "Fedu": 3,
        "schoolsup": "yes",
        "famsup": "yes",
        "higher": "yes",
        "internet": "yes",
        "health": 4,
    },
    {
        "studytime": 2,
        "failures": 0,
        "absences": 5,
        "Medu": 2,
        "Fedu": 2,
        "schoolsup": "no",
        "famsup": "yes",
        "higher": "yes",
        "internet": "yes",
        "health": 3,
    },
]

print("=" * 65)
print("STUDENT PERFORMANCE PREDICTION DIAGNOSTICS")
print("=" * 65)

for index, case in enumerate(test_cases, start=1):
    input_df = pd.DataFrame([case])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    print(f"\nTest Case {index}")
    print("-" * 30)
    print("Prediction:", prediction)

    for class_name, probability in zip(
        model.classes_, probabilities
    ):
        print(f"{class_name}: {probability * 100:.2f}%")