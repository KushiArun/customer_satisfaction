
# 📊 Customer Satisfaction Prediction

This project predicts **customer satisfaction** from support ticket data using **Machine Learning**.
The dataset includes details about customers, products, support tickets, and satisfaction ratings.

---

## 🚀 Features

* **Data Preprocessing**: Handles missing values, encodes categorical variables, extracts features (year, month, age groups).
* **Exploratory Analysis**: Ticket trends, customer demographics, and support channel insights.
* **ML Models Implemented**:

  * Random Forest
  * Logistic Regression
  * Support Vector Machine (SVM)
  * Gradient Boosting
* **Multi-class Prediction (1–5 stars)** and **Binary Classification (Satisfied vs Not Satisfied)**.
* **Hyperparameter Tuning** with GridSearchCV.
* **Feature Importance Analysis** with visualization.

---

## 🗂 Dataset

* Source: Provided as `customer_support_tickets.csv`
* Features include:

  * Customer demographics (age, gender)
  * Product purchased & purchase date
  * Ticket type, channel, priority, status
  * Response time & resolution time
  * **Customer Satisfaction Rating** (target variable)

---

## ⚙️ Tech Stack

* **Language**: Python
* **Libraries**:

  * `pandas`, `numpy` (data processing)
  * `matplotlib`, `seaborn` (visualization)
  * `scikit-learn` (ML models & preprocessing)

---

## 🏗️ Project Workflow

1. Load and clean dataset
2. Encode categorical variables & engineer features
3. Train/test split & scaling
4. Train multiple ML models
5. Evaluate performance (accuracy, precision, recall, F1)
6. Tune Random Forest with GridSearchCV
7. Visualize feature importance

---

## 📈 Results

* **Multi-class prediction (1–5 ratings)** → \~20% accuracy (difficult due to imbalance).
* **Binary classification (Satisfied vs Not)** → \~62% accuracy with tuned Random Forest.
* **Key insights**: Ticket Priority, Response Time, and Ticket Type are top predictors of satisfaction.

---

## 📊 Example Output

```text
🔹 Multi-class Classification
Random Forest Accuracy: 0.22
Logistic Regression Accuracy: 0.19
SVM Accuracy: 0.19
Gradient Boosting Accuracy: 0.21

🔹 Binary Classification (Satisfied vs Not)
Best Random Forest Accuracy: 0.62
```

Feature importance plot is saved as **`feature_importances.png`**.
feature_importances.png

---


