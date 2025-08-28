
-----

### 🏀 Basketball High Scorer Prediction

This application aims to predict whether a basketball player will score over 20 points per game, based on their in-game statistics.

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


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Dataset ---
print("--- 1. Load Dataset ---")
try:
    # IMPORTANT: You need to download an NBA player stats dataset from Kaggle.
    # A common dataset for this purpose is 'NBA Player Stats 2017-2018' or similar.
    # Make sure the CSV file is in the same directory as this script.
    # Replace 'nba_player_stats.csv' with your actual downloaded file name.
    df = pd.read_csv('nba_player_stats.csv') # Example filename
    print("NBA Player Stats dataset loaded successfully.")
    print("First 5 rows of the data:")
    print(df.head())
    print("\nDataset Info (dtype and non-null counts):")
    df.info()
except FileNotFoundError:
    print("Error: 'nba_player_stats.csv' file not found.")
    print("Please download an NBA player statistics dataset from Kaggle (e.g., 'NBA Player Stats 2017-2018')")
    print("and place the CSV file in the same directory as this script, then update the filename in the code.")
    print("Example Kaggle search: https://www.kaggle.com/datasets?search=nba+player+stats")
    exit()

# --- 2. Feature Engineering & Target Variable Definition ---
print("\n--- 2. Feature Engineering & Target Variable Definition ---")

# Define the target variable: 1 if 'PTS' (Points) > 20, 0 otherwise
# Make sure 'PTS' column exists in your dataset. Adjust column name if needed.
if 'PTS' not in df.columns:
    print("Error: 'PTS' column (Points per game) not found in the dataset.")
    print("Please check your dataset's column names and adjust 'PTS' if it's different (e.g., 'PPG').")
    exit()

df['high_scorer'] = df['PTS'].apply(lambda x: 1 if x > 20 else 0)
print(f"Distribution of 'high_scorer' target variable:\n{df['high_scorer'].value_counts()}")

# Select features (excluding player name, team, season, and original 'PTS' column)
# You'll need to inspect your specific dataset's columns.
# Common features might include: 'AST', 'REB', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'MIN', etc.
# Adjust the list of features to exclude/include based on your dataset.
features_to_exclude = ['PLAYER', 'TEAM', 'SEASON', 'PTS', 'high_scorer'] # Adjust as per your dataset
# Automatically select numerical features, then filter out excluded ones
X = df.select_dtypes(include=np.number).drop(columns=[col for col in features_to_exclude if col in df.columns], errors='ignore')
y = df['high_scorer']

print(f"\nFeatures (X) selected. Sample head:\n{X.head()}")
print(f"Target (y) head:\n{y.head()}")

# Drop any rows where target is NaN (shouldn't happen if 'PTS' exists, but good practice)
X = X[y.notna()]
y = y[y.notna()]

# --- 3. Data Splitting ---
print("\n--- 3. Data Splitting ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# --- 4. Building Pipelines for Models ---
print("\n--- 4. Building Pipelines for Models ---")

# Pipeline for RandomForestClassifier
# Steps: Impute missing values -> Scale features -> Train RandomForest
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # Handle missing values by imputing with the mean
    ('scaler', StandardScaler()),                 # Scale features to have zero mean and unit variance
    ('classifier', RandomForestClassifier(random_state=42)) # RandomForest model
])
print("RandomForestClassifier Pipeline created.")

# Pipeline for LogisticRegression
# Steps: Impute missing values -> Scale features -> Train LogisticRegression
lr_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # Handle missing values by imputing with the mean
    ('scaler', StandardScaler()),                 # Scale features
    ('classifier', LogisticRegression(random_state=42, max_iter=1000)) # Logistic Regression model
])
print("LogisticRegression Pipeline created.")

# --- 5. Model Training and Evaluation ---
print("\n--- 5. Model Training and Evaluation ---")

# --- Model 1: RandomForestClassifier ---
print("\n--- Training and Evaluating RandomForestClassifier ---")
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

print("\nAccuracy Score (RandomForest):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

print("\nClassification Report (RandomForest):")
print(classification_report(y_test, y_pred_rf, target_names=['Not High Scorer', 'High Scorer']))

print("\nConfusion Matrix (RandomForest):")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Not High Scorer', 'High Scorer'], yticklabels=['Not High Scorer', 'High Scorer'])
plt.title('Confusion Matrix - RandomForestClassifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Cross-validation for RandomForest
print("\n--- Cross-validation for RandomForestClassifier ---")
# Use the full X and y for cross-validation on the pipeline
cv_scores_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation scores (5-fold): {cv_scores_rf}")
print(f"Average Cross-validation score: {cv_scores_rf.mean():.4f}")


# --- Model 2: LogisticRegression ---
print("\n--- Training and Evaluating LogisticRegression ---")
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

print("\nAccuracy Score (LogisticRegression):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

print("\nClassification Report (LogisticRegression):")
print(classification_report(y_test, y_pred_lr, target_names=['Not High Scorer', 'High Scorer']))

print("\nConfusion Matrix (LogisticRegression):")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Not High Scorer', 'High Scorer'], yticklabels=['Not High Scorer', 'High Scorer'])
plt.title('Confusion Matrix - LogisticRegression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Cross-validation for Logistic Regression
print("\n--- Cross-validation for LogisticRegression ---")
# Use the full X and y for cross-validation on the pipeline
cv_scores_lr = cross_val_score(lr_pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation scores (5-fold): {cv_scores_lr}")
print(f"Average Cross-validation score: {cv_scores_lr.mean():.4f}")

print("\n--- Basketball High Scorer Prediction process completed ---")
```

-----

### How to Use the Application:

1.  **Download Dataset:**

      * You need to obtain an NBA player statistics dataset from Kaggle. There are many options, search for "NBA player stats" (e.g., "NBA Player Stats 2017-2018", "All NBA Stats 1996-2022").
      * **Crucially**, ensure your downloaded CSV file contains a column for **Points Per Game**, often named `'PTS'` or `'PPG'`. If it's named differently, you'll need to adjust the code where `'PTS'` is referenced.
      * Place the downloaded CSV file (e.g., `nba_player_stats.csv`) in the same directory as your Python script.

2.  **Install Libraries:**

      * Ensure you have all the necessary libraries installed. If not, install them using pip:
        ```bash
        pip install pandas scikit-learn matplotlib seaborn numpy
        ```

3.  **Run the Script:**

      * Open your terminal or command prompt, navigate to the directory where you saved the script, and run it:
        ```bash
        python your_script_name.py
        ```

### Key Concepts in this Application:

  * **Target Variable:** We create a binary target variable, `high_scorer`, which is `1` if a player's average points per game (`PTS`) is greater than 20, and `0` otherwise. This transforms the problem into a binary classification task.
  * **Feature Selection:** We select relevant numerical features (like assists, rebounds, steals, blocks, shooting percentages, minutes played, etc.) while excluding identifying columns (player name, team) and the original `PTS` column to prevent data leakage.
  * **Handling Missing Data (`SimpleImputer`):** NBA datasets often have missing values (e.g., for players who played very few games). `SimpleImputer(strategy='mean')` fills these missing numerical values with the mean of their respective columns.
  * **Feature Scaling (`StandardScaler`):** Features like points, assists, or rebounds can have different scales. `StandardScaler` transforms them to have a mean of 0 and a standard deviation of 1, which helps algorithms like Logistic Regression perform better.
  * **Pipelines (`Pipeline`):** Scikit-learn's `Pipeline` object is incredibly useful for chaining multiple processing steps (imputation, scaling) with an estimator (classifier). This ensures that data preprocessing is applied consistently during both training and cross-validation, preventing data leakage from the test set into the training phase.
  * **Algorithms:**
      * **Random Forest Classifier:** An ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting.
      * **Logistic Regression:** A linear model used for binary classification, which models the probability of a binary outcome.
  * **Evaluation:**
      * **`accuracy_score`:** Measures the overall correctness of the predictions.
      * **`classification_report`:** Provides precision, recall, and F1-score for each class ("Not High Scorer" and "High Scorer"), which are essential metrics, especially for imbalanced datasets.
      * **`confusion_matrix`:** Visualizes the counts of true positives, true negatives, false positives, and false negatives, helping to understand where the model makes errors.
      * **`cross_val_score`:** Performs K-fold cross-validation (here, 5-fold) on the entire pipeline. This gives a more reliable estimate of the model's performance on unseen data by training and testing on different subsets of the data multiple times.

