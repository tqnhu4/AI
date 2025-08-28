
-----

### 🎓 Student Performance Classification Application

### Environment Setup

1.  **Clone this repository (if applicable)** or create the folder structure as above.
2.  **Create a virtual environment (recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```
3.  **Install required libraries**:
    ```bash
    pip install -r requirements.txt
    ```


This application predicts whether a student will "Pass" or "Fail" a course based on their personal and academic information.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Data Loading and Initial Exploration ---
print("--- 1. Data Loading and Initial Exploration ---")
try:
    # Load the Math course dataset. If you want Portuguese, change to 'student-por.csv'
    df = pd.read_csv('student-mat.csv', sep=';')
    print("Data loaded successfully.")
    print("First 5 rows of the data:")
    print(df.head())
    print("\nData Info (dtype and non-null counts):")
    df.info()
except FileNotFoundError:
    print("Error: 'student-mat.csv' file not found.")
    print("Please ensure you have downloaded and placed 'student-mat.csv' (or 'student-por.csv') in the same directory as this script.")
    print("You can download the dataset from: https://archive.ics.uci.edu/ml/datasets/student+performance")
    exit() # Exit if the file is not found

# --- 2. Feature Engineering and Target Variable Definition ---
print("\n--- 2. Feature Engineering and Target Variable Definition ---")
# Create the 'pass_fail' target variable: 1 if G3 >= 10 (Pass), 0 if G3 < 10 (Fail)
df['pass_fail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
print(f"Distribution of the 'pass_fail' target variable:\n{df['pass_fail'].value_counts()}")

# Identify categorical and numerical columns
# G1, G2, G3 are excluded as they are grades and would cause data leakage
features_to_exclude = ['G1', 'G2', 'G3', 'pass_fail']
# Automatically identify categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
# Automatically identify numerical columns (after excluding target and explicitly categorical ones)
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in features_to_exclude]

print(f"\nIdentified categorical columns: {categorical_cols}")
print(f"Identified numerical columns (excluded): {numerical_cols}")

# --- 3. Data Preprocessing ---
print("\n--- 3. Data Preprocessing ---")

# Label Encoding for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Encoded column '{col}' to numerical format.")

# Split data into training and testing sets
X = df.drop(features_to_exclude, axis=1) # Features
y = df['pass_fail'] # Target variable

# Check if X and y are empty
if X.empty or y.empty:
    print("Error: Features or target variable are empty after preprocessing. Please check your data and column selection.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTraining set size: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set size: X_test={X_test.shape}, y_test={y_test.shape}")

# Feature Scaling (Standardization) for numerical columns (important for KNN)
# Apply only to the selected numerical columns
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print("Standardized numerical columns.")

# --- 4. Model Training and Evaluation ---
print("\n--- 4. Model Training and Evaluation ---")

# --- Model 1: Logistic Regression ---
print("\n--- Training Logistic Regression ---")
lr_model = LogisticRegression(random_state=42, max_iter=1000) # Increase max_iter to ensure convergence
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n--- Evaluating Logistic Regression ---")
print("Confusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Fail', 'Pass']))

# Cross-validation for Logistic Regression
print("\n--- Cross-validation for Logistic Regression ---")
cv_scores_lr = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores (5-fold): {cv_scores_lr}")
print(f"Average Cross-validation score: {cv_scores_lr.mean():.4f}")

# --- Model 2: K-Nearest Neighbors (KNN) ---
print("\n--- Training K-Nearest Neighbors (KNN) ---")
# You can experiment with different 'n_neighbors' values
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("\n--- Evaluating K-Nearest Neighbors (KNN) ---")
print("Confusion Matrix:")
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.title('Confusion Matrix - KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn, target_names=['Fail', 'Pass']))

# Cross-validation for KNN
print("\n--- Cross-validation for K-Nearest Neighbors (KNN) ---")
cv_scores_knn = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores (5-fold): {cv_scores_knn}")
print(f"Average Cross-validation score: {cv_scores_knn.mean():.4f}")

print("\n--- Student Performance Classification process completed ---")
```

-----

### How to Use the Application:

1.  **Download Dataset:** Go to the [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance) and download the `student.zip` file. Extract its contents.
2.  **Place Data File:** Put the `student-mat.csv` (for Math course) or `student-por.csv` (for Portuguese language course) file in the same directory as your Python script. This script defaults to using `student-mat.csv`. If you want to use the Portuguese data, change the line `pd.read_csv('student-mat.csv', sep=';')` to `pd.read_csv('student-por.csv', sep=';')`.
3.  **Install Libraries:** Ensure you have all the necessary libraries installed. If not, install them using pip:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn numpy
    ```
4.  **Run the Script:** Open your terminal or command prompt, navigate to the directory where you saved the script, and run it:
    ```bash
    python your_script_name.py
    ```
