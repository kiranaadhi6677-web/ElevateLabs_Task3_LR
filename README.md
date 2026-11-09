# Diabetes Progression Prediction Using Linear Regression

## AI & ML Internship – Task 3

### 1. Project Overview

This project implements Simple Linear Regression and Multiple Linear Regression using Python and Machine Learning libraries.

The objective is to predict diabetes progression based on different input features and understand regression modeling, model evaluation, and visualization.

### 2. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

### 3. Dataset Information

The project uses the Diabetes dataset available in Scikit-learn.

The dataset contains:
- 442 observations.
- 10 independent features.
- 1 target variable representing diabetes disease progression.

### 4. Project Implementation

The following steps were performed:

1. Dataset loading.
2. Data preprocessing.
3. Missing value checking.
4. Train-test splitting.
5. Simple Linear Regression using BMI.
6. Multiple Linear Regression using all 10 features.
7. Model training and prediction.
8. Evaluation using MAE, MSE and R² Score.
9. Regression visualization.
10. Model performance comparison.

### 5. Model Performance

| Evaluation Metric | Simple Linear Regression | Multiple Linear Regression |
|---|---:|---:|
| MAE | 52.2599 | 42.7941 |
| MSE | 4061.8259 | 2900.1936 |
| R² Score | 0.2334 | 0.4526 |

### 6. Result and Conclusion

Both regression models were successfully implemented.

Multiple Linear Regression performed better than Simple Linear Regression on the test dataset.

It achieved:
- Lower Mean Absolute Error.
- Lower Mean Squared Error.
- Higher R² Score.

This demonstrates how using multiple independent variables can improve regression performance.

### 7. Project Structure

```text
Elevatelabs_Task3_LR/
│
├── main.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── diabetes_dataset.csv
│
└── outputs/
    ├── simple_linear_regression.png
    ├── actual_vs_predicted.png
    └── model_comparison.csv
```

### 8. How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

### 9. Learning Outcomes

- Understanding Simple Linear Regression.
- Understanding Multiple Linear Regression.
- Dataset preprocessing.
- Model training and testing.
- Regression evaluation metrics.
- Model coefficient interpretation.
- Data visualization.

### 10. Author

AADITHYA

AI & ML Internship