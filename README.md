# 🎓 Student Performance Classifier

An interactive machine learning application that estimates student academic performance using study habits, previous class failures, attendance, and other student-related factors.

The application classifies predicted performance into three categories:

* **Low**
* **Average**
* **High**

It also provides estimated prediction probabilities, academic improvement suggestions, feature importance information, and a model evaluation dashboard.

> **Disclaimer:** This is an experimental educational project. Its predictions are not definitive assessments of students and should not be used to make consequential academic decisions.

---

## 🚀 Features

* Student performance classification using machine learning
* Interactive Streamlit web application
* Estimated prediction probability
* Feature importance visualization
* Personalized, rule-based improvement suggestions
* Model evaluation dashboard
* Confusion matrix visualization
* Accuracy, balanced accuracy, precision, recall, and Macro F1-score
* Experiments comparing multiple classification and regression models
* Cross-validation and feature analysis

---

## 🛠️ Tech Stack

| Technology   | Purpose                                       |
| ------------ | --------------------------------------------- |
| Python       | Core programming language                     |
| Pandas       | Dataset loading and manipulation              |
| NumPy        | Numerical operations                          |
| Scikit-learn | Preprocessing, model training, and evaluation |
| Matplotlib   | Visualizations                                |
| Seaborn      | Statistical charts                            |
| Joblib       | Saving and loading the trained model          |
| Streamlit    | Interactive web application                   |

---

## 📊 Dataset

This project uses the **UCI Student Performance Dataset**, specifically the mathematics student dataset.

Dataset: https://archive.ics.uci.edu/dataset/320/student%2Bperformance

The dataset contains information about students' academic performance, study habits, family background, school support, absences, and other factors.

The target variable is derived from the final mathematics grade, `G3`.

### Performance categories

| Final Grade (`G3`) | Category |
| ------------------ | -------- |
| 0–9                | Low      |
| 10–14              | Average  |
| 15–20              | High     |

The model excludes `G1` and `G2` because the project focuses on early performance estimation without using the first- and second-period grades.

---

## 🧠 Machine Learning Workflow

1. Load the dataset.
2. Create performance categories from the final grade.
3. Select the input features.
4. Split the data into training and test sets.
5. Preprocess categorical variables using One-Hot Encoding.
6. Train and compare machine learning models.
7. Evaluate models using cross-validation and classification metrics.
8. Analyze feature importance and model limitations.
9. Save the selected model.
10. Use the trained model in the Streamlit application.

### Input features

* Weekly study time
* Previous class failures
* Number of absences
* Mother's education level
* Father's education level
* Extra school support
* Family educational support
* Intention to pursue higher education
* Internet access at home
* Current health rating

### Model experiments

The project explores several algorithms, including:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* Dummy Classifier as a baseline

Regression experiments were also conducted to investigate the prediction of numerical final grades.

---

## 📈 Model Evaluation

The project compares multiple classification algorithms using five-fold stratified cross-validation.

### Cross-validation results

| Model               | Mean Accuracy | Mean Balanced Accuracy | Mean Macro F1 |
| ------------------- | ------------: | ---------------------: | ------------: |
| Random Forest       |        48.43% |                 46.52% |        45.67% |
| Logistic Regression |        43.02% |                 48.07% |        43.25% |

Random Forest achieved the highest mean accuracy and Macro F1-score in this experiment, while Logistic Regression achieved higher balanced accuracy.

Based on these results, Random Forest remains the primary model for the application.

### Saved model evaluation

The currently saved Random Forest model achieved the following results on the development test split:

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          | 41.77% |
| Balanced Accuracy | 39.72% |
| Macro F1-score    | 39.49% |

The saved model's test performance is lower than its cross-validation results. These metrics should be interpreted cautiously because the same development split has been examined repeatedly during experimentation.

The evaluation dashboard displays the confusion matrix, class-wise precision, recall, and F1-score.

### Limitations

* Predictive performance remains modest.
* The model does not identify all performance categories equally well.
* The dataset contains fewer High-category examples than Average-category examples.
* Predicted probabilities are not necessarily calibrated.
* Further validation on an untouched dataset is needed before claiming generalization to new students.


## 💻 Installation and Setup

### 1. Clone or download the project

Download the repository or extract the project ZIP file.

### 2. Open the project directory

```bash
cd student-performance-classifier
```

### 3. Create a virtual environment

Windows:

```powershell
python -m venv acm_env
```

Activate it:

```powershell
.\acm_env\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the Python executable inside the environment directly.

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the prediction application

```bash
python -m streamlit run app.py
```

### 6. Open the model evaluation page

Once the application opens, select the **Model Evaluation** page from the Streamlit sidebar.

---

## 📁 Project Structure

```text
student-performance-classifier/
│
├── app.py
├── pages/
│   └── 2_Model_Evaluation.py
│
├── train_model.py
├── validate_models.py
├── tune_model.py
├── tune_logistic.py
├── feature_analysis.py
├── feature_importance.py
├── compare_features.py
├── train_regression.py
├── evaluate_saved_model.py
├── check_model.py
│
├── student-mat.csv
├── student_performance_model.pkl
├── requirements.txt
├── README.md
│
└── visualizations/
```

The directory may contain additional experimental scripts and generated results.

---

## 🔍 Limitations and Future Improvements

### Current limitations

* Predictive performance is limited with the selected input features.
* The dataset is relatively small.
* The model's performance differs across performance categories.
* Estimated probabilities may not be calibrated.
* Feature importance does not establish causation.
* The model has not been validated for real-world academic decision-making.

### Future improvements

* Evaluate the model on an independent dataset.
* Investigate additional early-available predictive features.
* Improve minority-class identification.
* Assess probability calibration.
* Add interactive visualizations and downloadable reports.
* Conduct fairness and robustness analysis.
* Compare results across repeated data splits.

---

## 🎯 Project Objective

The goal is to explore how machine learning can support early understanding of student academic performance while demonstrating responsible data preprocessing, model evaluation, visualization, and application development.

This project is intended for educational and experimental purposes, not as a replacement for teachers' judgment or individual student support.
