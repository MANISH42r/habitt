# 🚀 Productivity Prediction System (ML Project)

## 📌 Overview

This project is an **end-to-end Machine Learning system** that predicts whether a person is **productive or not** based on their daily habits.

It also provides **personalized insights** to improve productivity using rule-based logic and model explainability.

---

## 🎯 Features

* 🤖 Predicts productivity (Productive / Not Productive)
* 📊 Multiple ML models comparison:

  * Logistic Regression
  * Random Forest
  * Gradient Boosting
* 📈 Data visualization (distribution & correlation)
* 📉 Confusion matrix & evaluation metrics
* 🔍 Feature importance analysis
* 🧠 SHAP explainability for model interpretation
* 💡 Personalized habit-based insights
* 💾 Model saving for deployment

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* SHAP
* Joblib

---

## 📂 Project Workflow

### 1️⃣ Data Loading

* Dataset loaded from Excel file
* Initial inspection of data shape and structure

### 2️⃣ Data Visualization

* Productivity score distribution
* Correlation heatmap

### 3️⃣ Data Preprocessing

* Feature-target separation
* Train-test split
* Feature scaling (for Logistic Regression)

### 4️⃣ Model Training

* Logistic Regression (scaled data)
* Random Forest (main model)
* Gradient Boosting

### 5️⃣ Model Evaluation

* Accuracy comparison
* Classification report
* Confusion matrix

### 6️⃣ Feature Importance

* Random Forest feature importance visualization

### 7️⃣ Explainability (SHAP)

* Beeswarm plot
* Bar plot
* Waterfall plot

### 8️⃣ Prediction System

* Takes user input
* Predicts productivity
* Shows confidence score

### 9️⃣ Rule-Based Insights

Provides suggestions like:

* Improve sleep
* Reduce screen time
* Increase study hours
* Exercise more

---

## 📊 Model Output

* ✅ Productive
* ❌ Not Productive
* 📈 Confidence Score (%)
* 🧠 Personalized Insights

---

## 💾 Saved Files

```id="f1"
productivity_model.pkl     # Trained Random Forest model
feature_order.pkl         # Feature order for prediction
```

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash id="f2"
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib openpyxl
```

### 2️⃣ Run Script

```bash id="f3"
python your_script_name.py
```

---

## 📌 Important Notes

* Dataset file must be present:

```id="f4"
habit_productivity_dataset_v2 (1).xlsx
```

* Feature order is maintained using:

```id="f5"
feature_order.pkl
```

* Input data must match training features

---

## 🔮 Future Improvements

* Add Streamlit UI
* Deploy as web app
* Improve model accuracy
* Add real-time tracking
* Store user data in database

---

## 👨‍💻 Author

**Manish**
3rd Year CSE (AI & ML)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it 🚀
