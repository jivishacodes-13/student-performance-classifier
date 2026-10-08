# 🎓 Student Performance Classifier

An AI/ML-based web application that predicts a student's academic performance as **High, Average, or Low** based on academic, behavioral, and student-related factors.

The application also provides a **confidence score, academic risk assessment, top contributing factors, and personalized improvement suggestions**.

---

## 🚀 Features

- 📊 Predicts student performance:
  - High
  - Average
  - Low
- 🎯 Displays model confidence score
- 🔍 Shows the top factors influencing the prediction
- ⚠️ Provides an academic risk assessment
- 💡 Generates improvement suggestions
- 🖥️ Interactive Streamlit interface
- 🌐 Easy to run locally

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** – Data loading and manipulation
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine learning and preprocessing
- **Joblib** – Model serialization
- **Streamlit** – Interactive web application

---

## 📂 Dataset

The project uses the **UCI Student Performance Dataset**.

Dataset source:

https://archive.ics.uci.edu/dataset/320/student%2Bperformance

The project uses the `student-mat.csv` dataset containing student information and final mathematics grades.

The original dataset contains features related to:

- Study time
- Past failures
- Absences
- Parents' education
- School support
- Family support
- Higher education goals
- Internet access
- Health

The final grade (`G3`) is used to create the performance classes.

---

## 🧠 Performance Classification

The final grade is converted into three performance categories:

| Final Grade (G3) | Performance |
|---|---|
| 15–20 | High |
| 10–14 | Average |
| 0–9 | Low |

`G1` and `G2` were excluded from the model inputs because they are strongly correlated with the final grade and could cause target leakage.

---

## 🤖 Machine Learning Approach

### 1. Data Preprocessing

The dataset is loaded using Pandas with `;` as the separator.

Categorical features are converted using:

- `OneHotEncoder`

Numerical features are passed directly to the model.

### 2. Selected Features

The model uses:

- Study time
- Past class failures
- Number of absences
- Mother's education
- Father's education
- School support
- Family educational support
- Higher education goal
- Internet access
- Health

### 3. Model

A **Random Forest Classifier** is used for classification.

Configuration:

- 200 decision trees
- Random state = 42
- Balanced class weights

### 4. Evaluation

The model is evaluated using:

- Accuracy
- Classification report
- Confusion matrix

Feature importance from the Random Forest model is also used to display the most influential factors in the application.

---

## 📁 Project Structure

```text
student-performance-classifier/
│
├── app.py
├── train_model.py
├── student_performance_model.pkl
├── student-mat.csv
├── requirements.txt
└── README.md