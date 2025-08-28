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