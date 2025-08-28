
-----

## 1\. 📊 Project: Housing Price Prediction

### Objective

The main goal of this project is to build a **Linear Regression** model capable of predicting house prices based on various characteristics of the house and its surrounding area. We'll use the **California Housing** dataset available in the `sklearn` library.

### Steps to Implement

We will go through the following steps in building the model:

1.  **Load and Explore Data**: Load the California Housing dataset and examine its structure, features, and target variable.
2.  **Split Data**: Separate the data into training and testing sets to fairly evaluate the model's performance.
3.  **Data Preprocessing**: Standardize features using `StandardScaler` to ensure all features are on the same scale, helping the model learn more effectively.
4.  **Build and Train Model**: Initialize and train the Linear Regression model on the training dataset.
5.  **Evaluate Model**: Use the testing dataset to assess the model's performance by calculating the **Mean Squared Error (MSE)**.

### Environment Setup

1.  **Clone this repository (if applicable)** or create the folder structure as above.
2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```
3.  **Install required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

### Library Requirements

Make sure you have the following libraries installed:

  * `scikit-learn`
  * `numpy`
  * `pandas`

You can install them by running the following command in your terminal or Anaconda Prompt:
`pip install scikit-learn numpy pandas`

-----

### Project Code Implementation

Here's the complete Python code for you to implement this project:

```python
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("Starting the Housing Price Prediction project...")
print("-" * 50)

# 1. Load and explore data
print("1. Loading and exploring data...")
housing = fetch_california_housing(as_frame=True)
X = housing.data  # Features
y = housing.target # Target (house price)

print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print("\nFirst 5 rows of feature data (X):")
print(X.head())
print("\nDescriptive statistics of features (X):")
print(X.describe())
print("\nFirst 5 values of the target variable (y - house price):")
print(y.head())
print("-" * 50)

# 2. Split data into training and testing sets
print("2. Splitting data into training and testing sets (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size (X_train): {X_train.shape}")
print(f"Test set size (X_test): {X_test.shape}")
print("-" * 50)

# 3. Data Preprocessing: Standardize features
print("3. Data Preprocessing: Standardizing features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optionally, convert back to DataFrame for better readability
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

print("First 5 rows of scaled training data (X_train_scaled):")
print(X_train_scaled_df.head())
print("-" * 50)

# 4. Build and train the Linear Regression model
print("4. Building and training the Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")
print(f"Model Coefficients:\n{model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print("-" * 50)

# 5. Evaluate the model
print("5. Evaluating the model on the test set...")
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # RMSE is easier to interpret as it's in the same unit as the target variable

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("-" * 50)

print("\nHousing Price Prediction project completed!")
print("Your model can now predict house prices based on the given features.")
```

-----

### Code Explanation

  * **`fetch_california_housing(as_frame=True)`**: Loads the California Housing dataset. `as_frame=True` ensures the data is returned as a Pandas DataFrame, making it easier to manipulate.
  * **`train_test_split(X, y, test_size=0.2, random_state=42)`**: This function splits the data into 80% for training and 20% for testing. `random_state` ensures consistent splitting results every time you run the code.
  * **`StandardScaler()`**: Initializes an object to standardize the data.
      * `fit_transform(X_train)`: Learns parameters (mean and standard deviation) from `X_train` and applies the transformation to `X_train`.
      * `transform(X_test)`: Only applies the transformation to `X_test` using the parameters learned from `X_train`. It's crucial not to `fit` `X_test` again to prevent data leakage from the test set.
  * **`LinearRegression()`**: Initializes the linear regression model.
  * **`model.fit(X_train_scaled, y_train)`**: Trains the model using the standardized training data. The model learns the relationship between the features and house prices.
  * **`model.predict(X_test_scaled)`**: Uses the trained model to predict house prices on the test dataset.
  * **`mean_squared_error(y_test, y_pred)`**: Calculates the MSE between the actual values (`y_test`) and the predicted values (`y_pred`). MSE indicates the average magnitude of prediction errors. A smaller MSE value means a better model. **RMSE** (Root Mean Squared Error) is often preferred over MSE because it's in the same unit as the target variable, making it easier to interpret.

