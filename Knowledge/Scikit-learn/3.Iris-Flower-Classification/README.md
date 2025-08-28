

### 🦋 **Iris Flower Classification**

This application aims to classify three species of Iris flowers (Setosa, Versicolor, Virginica) based on the measurements of their sepal length, sepal width, petal length, and petal width.

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


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. Load Dataset ---
print("--- 1. Load Dataset ---")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

print("Iris dataset loaded successfully.")
print("Features (X) head:")
print(X.head())
print("\nTarget (y) head:")
print(y.head())
print(f"\nTarget names: {iris.target_names}")
print(f"Dataset shape: {X.shape}")

# --- 2. Data Splitting ---
print("\n--- 2. Data Splitting ---")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # stratify for balanced classes
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# --- 3. Model Training and Hyperparameter Tuning with GridSearchCV ---

# --- Model 1: Decision Tree Classifier ---
print("\n--- 3a. Decision Tree Classifier with GridSearchCV ---")
# Define the parameter grid for Decision Tree
dt_param_grid = {
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_grid_search = GridSearchCV(dt_classifier, dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
dt_grid_search.fit(X_train, y_train)

print(f"\nBest parameters for Decision Tree: {dt_grid_search.best_params_}")
print(f"Best cross-validation score for Decision Tree: {dt_grid_search.best_score_:.4f}")

# Get the best Decision Tree model
best_dt_model = dt_grid_search.best_estimator_

# --- Model 2: Support Vector Machine (SVM) ---
print("\n--- 3b. Support Vector Machine (SVM) with GridSearchCV ---")
# Define the parameter grid for SVM
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
    'kernel': ['rbf', 'linear'] # RBF is common, linear is also a good option
}

svm_classifier = SVC(random_state=42)
svm_grid_search = GridSearchCV(svm_classifier, svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
svm_grid_search.fit(X_train, y_train)

print(f"\nBest parameters for SVM: {svm_grid_search.best_params_}")
print(f"Best cross-validation score for SVM: {svm_grid_search.best_score_:.4f}")

# Get the best SVM model
best_svm_model = svm_grid_search.best_estimator_

# --- 4. Model Evaluation ---
print("\n--- 4. Model Evaluation ---")

# --- Evaluate Decision Tree ---
print("\n--- Decision Tree Evaluation ---")
y_pred_dt = best_dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy Score: {dt_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))

print("\nConfusion Matrix:")
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)

# Plot Confusion Matrix for Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Visualize the Decision Tree (optional, requires graphviz or similar for more complex trees)
# For simple trees, plot_tree can be useful:
plt.figure(figsize=(15, 10))
plot_tree(best_dt_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()


# --- Evaluate SVM ---
print("\n--- Support Vector Machine (SVM) Evaluation ---")
y_pred_svm = best_svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy Score: {svm_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))

print("\nConfusion Matrix:")
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(cm_svm)

# Plot Confusion Matrix for SVM
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\n--- Iris Flower Classification process completed ---")
```

-----

### How to Use the Application:

1.  **Install Libraries:** Ensure you have all the necessary libraries installed. If not, install them using pip:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn numpy
    ```
2.  **Run the Script:** Open your terminal or command prompt, navigate to the directory where you saved the script, and run it:
    ```bash
    python your_script_name.py
    ```

### Explanation of the Code:

  * **1. Load Dataset:**

      * We use `sklearn.datasets.load_iris()` to directly load the Iris dataset, which is conveniently included with scikit-learn.
      * The features (measurements) are stored in `X` and the target (species labels) in `y`.

  * **2. Data Splitting:**

      * `train_test_split` divides the dataset into training (70%) and testing (30%) sets. `random_state` ensures reproducibility.
      * `stratify=y` is important for classification tasks to ensure that the proportion of target classes is roughly the same in both training and testing sets, preventing biased evaluation.

  * **3. Model Training and Hyperparameter Tuning with GridSearchCV:**

      * **Purpose of `GridSearchCV`:** This is a powerful tool for hyperparameter tuning. It exhaustively searches over specified parameter values for an estimator. For each combination of parameters, it performs cross-validation (here, 5-fold) and returns the best parameters found.
      * **Decision Tree:** We define a `param_grid` for `DecisionTreeClassifier` with different `max_depth`, `min_samples_split`, `min_samples_leaf`, and `criterion` options.
      * **Support Vector Machine (SVM):** For `SVC`, we explore different `C` (regularization parameter), `gamma` (kernel coefficient), and `kernel` types (`rbf` and `linear`).
      * `n_jobs=-1` tells `GridSearchCV` to use all available CPU cores, speeding up the process.

  * **4. Model Evaluation:**

      * For both models, after finding the best parameters using `GridSearchCV`, we use the `best_estimator_` to make predictions on the unseen `X_test` data.
      * **`accuracy_score`:** Provides the proportion of correctly classified instances.
      * **`classification_report`:** Offers a more detailed breakdown including precision, recall, and F1-score for each class, which is crucial for multi-class classification.
      * **`confusion_matrix`:** Shows the number of true positives, true negatives, false positives, and false negatives for each class. A heatmap visualization is included for better readability.
      * **Decision Tree Visualization (`plot_tree`):** For Decision Trees, `plot_tree` can be used to visualize the learned tree structure, though for deeper trees, this can become complex.
