from pathlib import Path

import pandas as pd

# -----------------------------------------
# 1. Load the dataset
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student-mat.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Make sure student-mat.csv is in the project folder."
    )

df = pd.read_csv(DATA_PATH, sep=";")

print("=" * 60)
print("STUDENT PERFORMANCE DATASET INSPECTION")
print("=" * 60)

# -----------------------------------------
# 2. Dataset dimensions
# -----------------------------------------

print("\n1. DATASET DIMENSIONS")
print("-" * 60)

print(f"Number of students: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# -----------------------------------------
# 3. Column names
# -----------------------------------------

print("\n2. COLUMN NAMES")
print("-" * 60)

print(df.columns.tolist())

# -----------------------------------------
# 4. First five records
# -----------------------------------------

print("\n3. FIRST FIVE RECORDS")
print("-" * 60)

print(df.head().to_string(index=False))

# -----------------------------------------
# 5. Data types
# -----------------------------------------

print("\n4. DATA TYPES")
print("-" * 60)

print(df.dtypes.to_string())

# -----------------------------------------
# 6. Missing values
# -----------------------------------------

print("\n5. MISSING VALUES")
print("-" * 60)

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("No missing values found.")

# -----------------------------------------
# 7. Duplicate records
# -----------------------------------------

print("\n6. DUPLICATE RECORDS")
print("-" * 60)

duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")

# -----------------------------------------
# 8. Summary statistics
# -----------------------------------------

print("\n7. NUMERICAL SUMMARY")
print("-" * 60)

print(df.describe().round(2).to_string())

# -----------------------------------------
# 9. Create target categories
# -----------------------------------------

print("\n8. PERFORMANCE CATEGORY DISTRIBUTION")
print("-" * 60)

def categorize_performance(grade):
    if grade >= 15:
        return "High"
    elif grade >= 10:
        return "Average"
    else:
        return "Low"

df["performance"] = df["G3"].apply(categorize_performance)

class_counts = df["performance"].value_counts()

print(class_counts.to_string())

print("\nClass percentages:")

class_percentages = (
    df["performance"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(class_percentages.to_string())

print("\nDataset inspection completed successfully!")