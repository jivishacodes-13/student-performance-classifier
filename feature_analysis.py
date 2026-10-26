from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")


# 1. Create performance categories
def classify_performance(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Average"
    return "Low"


df["Performance"] = df["G3"].apply(classify_performance)


# 2. Examine final grades by study time
print("=" * 55)
print("FINAL GRADE BY STUDY TIME")
print("=" * 55)

study_summary = (
    df.groupby("studytime")["G3"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

print(study_summary)


# 3. Examine final grades by previous failures
print("\nFINAL GRADE BY PAST FAILURES")
print("=" * 55)

failure_summary = (
    df.groupby("failures")["G3"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print(failure_summary)


# 4. Examine final grades by absences
print("\nABSENCES AND FINAL GRADES")
print("=" * 55)

print(
    df[["absences", "G3"]]
    .corr()
    .round(3)
)


# 5. Create output folder
output_dir = BASE_DIR / "visualizations"
output_dir.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# 6. Plot study time against final grade
fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="studytime",
    y="G3",
    ax=ax
)

ax.set_title("Study Time vs Final Mathematics Grade")
ax.set_xlabel("Weekly Study Time Category")
ax.set_ylabel("Final Grade (G3)")

fig.tight_layout()

fig.savefig(
    output_dir / "studytime_grade_analysis.png",
    dpi=300
)

plt.close(fig)


# 7. Plot previous failures against final grade
fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="failures",
    y="G3",
    ax=ax
)

ax.set_title("Past Failures vs Final Mathematics Grade")
ax.set_xlabel("Number of Past Class Failures")
ax.set_ylabel("Final Grade (G3)")

fig.tight_layout()

fig.savefig(
    output_dir / "failures_grade_analysis.png",
    dpi=300
)

plt.close(fig)


print("\nFeature analysis completed!")
print("Visualizations saved in:", output_dir)