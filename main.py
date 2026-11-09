# ==========================================
# LINEAR REGRESSION INTERNSHIP PROJECT
# Simple and Multiple Linear Regression
# ==========================================

# 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import os

# Create output folders
os.makedirs("dataset", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# 2. Load Dataset

print("\n========== DATASET INFORMATION ==========")

diabetes = load_diabetes()

df = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)

df["Target"] = diabetes.target

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Save dataset
df.to_csv("dataset/diabetes_dataset.csv", index=False)


# 3. Data Preprocessing

print("\n========== DATA PREPROCESSING ==========")

print("Missing values before cleaning:")
print(df.isnull().sum().sum())

df = df.dropna()

print("Missing values after cleaning:")
print(df.isnull().sum().sum())


# 4. SIMPLE LINEAR REGRESSION

print("\n========== SIMPLE LINEAR REGRESSION ==========")

# Selecting BMI as the only independent variable

X_simple = df[["bmi"]]
y = df["Target"]

# Split dataset into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X_simple,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create and train model

simple_model = LinearRegression()

simple_model.fit(X_train, y_train)

# Predictions

y_pred_simple = simple_model.predict(X_test)

# Evaluation

mae_simple = mean_absolute_error(y_test, y_pred_simple)

mse_simple = mean_squared_error(y_test, y_pred_simple)

r2_simple = r2_score(y_test, y_pred_simple)

print("\nSimple Linear Regression Results")

print("MAE:", mae_simple)
print("MSE:", mse_simple)
print("R2 Score:", r2_simple)

print("Coefficient:", simple_model.coef_[0])
print("Intercept:", simple_model.intercept_)


# 5. Plot Simple Linear Regression

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test["bmi"],
    y_test,
    label="Actual Values"
)

sorted_indices = np.argsort(X_test["bmi"].values)

plt.plot(
    X_test["bmi"].values[sorted_indices],
    y_pred_simple[sorted_indices],
    color="red",
    label="Regression Line"
)

plt.xlabel("BMI")
plt.ylabel("Diabetes Progression")
plt.title("Simple Linear Regression")
plt.legend()

plt.savefig("outputs/simple_linear_regression.png")

plt.show()


# 6. MULTIPLE LINEAR REGRESSION

print("\n========== MULTIPLE LINEAR REGRESSION ==========")

# Selecting multiple independent variables

X_multiple = df.drop("Target", axis=1)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multiple,
    y,
    test_size=0.2,
    random_state=42
)

# Create model

multiple_model = LinearRegression()

# Train model

multiple_model.fit(X_train_multi, y_train_multi)

# Predictions

y_pred_multi = multiple_model.predict(X_test_multi)


# 7. Evaluate Multiple Linear Regression

mae_multi = mean_absolute_error(
    y_test_multi,
    y_pred_multi
)

mse_multi = mean_squared_error(
    y_test_multi,
    y_pred_multi
)

r2_multi = r2_score(
    y_test_multi,
    y_pred_multi
)

print("\nMultiple Linear Regression Results")

print("MAE:", mae_multi)
print("MSE:", mse_multi)
print("R2 Score:", r2_multi)

print("Intercept:", multiple_model.intercept_)


# 8. Display Coefficients

print("\n========== MODEL COEFFICIENTS ==========")

coefficients = pd.DataFrame({
    "Feature": X_multiple.columns,
    "Coefficient": multiple_model.coef_
})

print(coefficients)


# 9. Compare Both Models

print("\n========== MODEL COMPARISON ==========")

comparison = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression"
    ],

    "MAE": [
        mae_simple,
        mae_multi
    ],

    "MSE": [
        mse_simple,
        mse_multi
    ],

    "R2 Score": [
        r2_simple,
        r2_multi
    ]
})

print(comparison)

comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# 10. Plot Actual vs Predicted Values

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test_multi,
    y_pred_multi
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted - Multiple Linear Regression")

plt.savefig("outputs/actual_vs_predicted.png")

plt.show()


# 11. Final Result

print("\n========== PROJECT COMPLETED ==========")

print("Simple Linear Regression completed successfully!")

print("Multiple Linear Regression completed successfully!")

print("Evaluation metrics calculated successfully!")

print("Graphs saved inside outputs folder!")

print("Dataset saved inside dataset folder!")
