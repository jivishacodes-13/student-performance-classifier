from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "student-mat.csv",
    sep=";"
)


def classify_performance(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Average"
    return "Low"


df["Performance"] = df["G3"].apply(
    classify_performance
)

# Count students in each category
counts = df["Performance"].value_counts()

# Calculate percentages
percentages = df["Performance"].value_counts(
    normalize=True
) * 100

print("=" * 50)
print("TARGET CLASS DISTRIBUTION")
print("=" * 50)

for category in ["Low", "Average", "High"]:
    print(
        f"{category}: "
        f"{counts.get(category, 0)} students "
        f"({percentages.get(category, 0):.2f}%)"
    )

print("\nTotal students:", len(df))