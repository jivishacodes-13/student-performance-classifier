from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# 1. Load dataset
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")

# ---------------------------------
# 2. Create performance categories
# ---------------------------------

def categorize_performance(grade):
    if grade >= 15:
        return "High"
    elif grade >= 10:
        return "Average"
    else:
        return "Low"

df["performance"] = df["G3"].apply(categorize_performance)

# Create output folder
output_dir = BASE_DIR / "visualizations"
output_dir.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# ---------------------------------
# GRAPH 1: Performance Distribution
# ---------------------------------

plt.figure(figsize=(8, 5))

order = ["Low", "Average", "High"]

sns.countplot(
    data=df,
    x="performance",
    hue="performance",
    order=order,
    hue_order=order,
    palette="viridis",
    legend=False
)

plt.title("Student Performance Distribution")
plt.xlabel("Performance--Category")
plt.ylabel("Number--of--Students")
plt.tight_layout()

plt.savefig(output_dir / "performance_distribution.png", dpi=300)
#plt.show()

# ---------------------------------
# GRAPH 2: Study Time vs Performance
# ---------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="studytime",
    hue="performance",
    order=[1, 2, 3, 4],
    hue_order=order,
    palette="viridis"
)

plt.title("Study Time vs Performance Category")
plt.xlabel("Weekly Study Time Category")
plt.ylabel("Number--of--Students")
plt.legend(title="Performance")
plt.tight_layout()

plt.savefig(output_dir / "studytime_vs_performance.png", dpi=300)
#plt.show()

# ---------------------------------
# GRAPH 3: Numerical Correlation Heatmap
# ---------------------------------

numerical_df = df.select_dtypes(include="number")

correlation = numerical_df.corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()

plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
#plt.show()

print("\nEDA completed successfully!")
print(f"Visualizations saved in: {output_dir}")
print("\nEDA completed successfully!")