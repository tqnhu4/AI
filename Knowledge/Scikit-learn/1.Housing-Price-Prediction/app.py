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