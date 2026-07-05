# 🌲 Driver Churn Prediction using Random Forest

This project uses a **Random Forest Classifier** to predict whether a driver is likely to leave the platform (churn) based on their activity and performance.

## 📌 Objective

Predict driver churn using historical driver information.

A prediction of:

- **0** → Driver stays
- **1** → Driver churns

---

## 📊 Dataset

| Feature | Description |
|----------|-------------|
| loads_completed | Total completed loads |
| last_login_days | Days since the driver's last login |
| rating | Driver rating |
| churn | Target variable (0 = Stay, 1 = Churn) |

Example dataset:

```csv
loads_completed,last_login_days,rating,churn
20,1,4.9,0
18,2,4.8,0
15,5,4.5,0
10,10,4.0,1
8,20,3.8,1
5,30,3.5,1
25,1,5.0,0
3,45,3.0,1
12,7,4.2,0
4,60,2.8,1
```

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-Learn

---

## 🌳 Machine Learning Algorithm

**Random Forest Classifier**

A Random Forest combines multiple Decision Trees and makes predictions based on the majority vote.

Benefits:

- Reduces overfitting
- More accurate than a single Decision Tree
- Handles complex relationships
- Provides feature importance

---

## 📁 Project Workflow

1. Load dataset
2. Select features and target
3. Split data into training and testing sets
4. Train a Random Forest model
5. Make predictions
6. Evaluate the model
7. Analyze feature importance

---

## 🎯 Features

```python
X = df[
    [
        "loads_completed",
        "last_login_days",
        "rating"
    ]
]
```

Target:

```python
y = df["churn"]
```

---

## 🚀 Model Training

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The model trains 100 decision trees and combines their predictions.

---

## 📈 Model Evaluation

Evaluation metrics used:

- Accuracy Score
- Confusion Matrix
- Classification Report

These metrics help determine how well the model predicts driver churn.

---

## 🔍 Feature Importance

Random Forest provides the importance of each feature.

Example:

```text
last_login_days    0.52
loads_completed    0.31
rating             0.17
```

Interpretation:

- Last login days is the strongest indicator of churn.
- Drivers inactive for a long time are more likely to leave.
- Loads completed and rating also influence the prediction.

---

## 🔮 Sample Prediction

Example:

```python
new_driver = pd.DataFrame({
    "loads_completed": [6],
    "last_login_days": [25],
    "rating": [3.6]
})

prediction = model.predict(new_driver)
```

Prediction:

```text
1
```

Meaning:

The model predicts the driver is likely to churn.

---

## 📂 Project Structure

```
driver-churn-random-forest/
│
├── driver_churn.csv
├── main.py
├── README.md
├── requirements.txt
└── venv/
```

---

## 📚 Concepts Learned

- Classification
- Ensemble Learning
- Random Forest
- Train/Test Split
- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance
- Predicting New Data

---

## 💡 Real-World Applications

Random Forest can be used for:

- Driver churn prediction
- Customer churn prediction
- Loan approval
- Fraud detection
- Disease diagnosis
- Employee attrition
- Credit risk assessment

---

## 🚀 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Larger datasets
- Model persistence using Joblib
- Deployment with FastAPI
- Model monitoring in production
