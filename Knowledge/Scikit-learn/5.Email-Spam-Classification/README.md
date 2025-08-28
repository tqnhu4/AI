
-----

### 💌 Email Spam Classification

This application aims to classify emails as "Spam" or "Not Spam" (Ham) based on their text content.

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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Dataset ---
print("--- 1. Load Dataset ---")
try:
    # IMPORTANT: Download the SMS Spam Collection Dataset from Kaggle.
    # The file is typically named 'spam.csv'. Place it in the same directory as this script.
    # The dataset has two columns: 'v1' (label: 'ham' or 'spam') and 'v2' (text message).
    df = pd.read_csv('spam.csv', encoding='latin-1')
    # Rename columns for clarity
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    # Drop unnecessary columns if they exist (common in this dataset)
    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

    print("SMS Spam Collection dataset loaded successfully.")
    print("First 5 rows of the data:")
    print(df.head())
    print("\nDataset Info (dtype and non-null counts):")
    df.info()
    print(f"\nDistribution of 'label':\n{df['label'].value_counts()}")

except FileNotFoundError:
    print("Error: 'spam.csv' file not found.")
    print("Please download the 'SMS Spam Collection Dataset' from Kaggle:")
    print("https://www.kaggle.com/uciml/sms-spam-collection-dataset")
    print("and place the 'spam.csv' file in the same directory as this script.")
    exit()

# --- 2. Data Preprocessing and Target Variable Definition ---
print("\n--- 2. Data Preprocessing and Target Variable Definition ---")

# Convert 'label' column to numerical format: 0 for 'ham', 1 for 'spam'
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
print(f"Distribution of 'label_encoded':\n{df['label_encoded'].value_counts()}")

X = df['text']            # Features (email/SMS text)
y = df['label_encoded']   # Target (0 for ham, 1 for spam)

# --- 3. Data Splitting ---
print("\n--- 3. Data Splitting ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# --- 4. Building Pipelines with Vectorizer and Multinomial Naive Bayes ---
print("\n--- 4. Building Pipelines with Vectorizer and Multinomial Naive Bayes ---")

# Pipeline with CountVectorizer
# Steps: Convert text to word counts -> Train Multinomial Naive Bayes
pipeline_cv = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
print("Pipeline with CountVectorizer and MultinomialNB created.")

# Pipeline with TfidfVectorizer
# Steps: Convert text to TF-IDF features -> Train Multinomial Naive Bayes
pipeline_tfidf = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])
print("Pipeline with TfidfVectorizer and MultinomialNB created.")

# --- 5. Hyperparameter Tuning and Model Training with GridSearchCV ---

# --- Model 1: CountVectorizer + Multinomial Naive Bayes ---
print("\n--- 5a. CountVectorizer + Multinomial Naive Bayes with GridSearchCV ---")

# Define the parameter grid for CountVectorizer + MultinomialNB
# For CountVectorizer: max_df, min_df, ngram_range
# For MultinomialNB: alpha (Laplace smoothing parameter)
param_grid_cv = {
    'vectorizer__max_df': [0.75, 1.0], # max_df=1.0 ignores words that appear in > 100% of documents (no limit)
    'vectorizer__min_df': [1, 5],      # min_df=1 includes words that appear in at least 1 document
    'vectorizer__ngram_range': [(1, 1), (1, 2)], # unigrams, or unigrams and bigrams
    'classifier__alpha': [0.1, 0.5, 1.0] # smoothing parameter
}

grid_search_cv = GridSearchCV(pipeline_cv, param_grid_cv, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_cv.fit(X_train, y_train)

print(f"\nBest parameters for CountVectorizer + MultinomialNB: {grid_search_cv.best_params_}")
print(f"Best cross-validation score for CountVectorizer + MultinomialNB: {grid_search_cv.best_score_:.4f}")

# Get the best model
best_model_cv = grid_search_cv.best_estimator_

# --- Model 2: TfidfVectorizer + Multinomial Naive Bayes ---
print("\n--- 5b. TfidfVectorizer + Multinomial Naive Bayes with GridSearchCV ---")

# Define the parameter grid for TfidfVectorizer + MultinomialNB
param_grid_tfidf = {
    'vectorizer__max_df': [0.75, 1.0],
    'vectorizer__min_df': [1, 5],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__alpha': [0.1, 0.5, 1.0]
}

grid_search_tfidf = GridSearchCV(pipeline_tfidf, param_grid_tfidf, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_tfidf.fit(X_train, y_train)

print(f"\nBest parameters for TfidfVectorizer + MultinomialNB: {grid_search_tfidf.best_params_}")
print(f"Best cross-validation score for TfidfVectorizer + MultinomialNB: {grid_search_tfidf.best_score_:.4f}")

# Get the best model
best_model_tfidf = grid_search_tfidf.best_estimator_

# --- 6. Model Evaluation ---
print("\n--- 6. Model Evaluation ---")

# --- Evaluate CountVectorizer + MultinomialNB ---
print("\n--- Evaluation: CountVectorizer + Multinomial Naive Bayes ---")
y_pred_cv = best_model_cv.predict(X_test)

cv_accuracy = accuracy_score(y_test, y_pred_cv)
print(f"Accuracy Score: {cv_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_cv, target_names=['Ham', 'Spam']))

print("\nConfusion Matrix:")
cm_cv = confusion_matrix(y_test, y_pred_cv)
print(cm_cv)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - CountVectorizer + MultinomialNB')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# --- Evaluate TfidfVectorizer + MultinomialNB ---
print("\n--- Evaluation: TfidfVectorizer + Multinomial Naive Bayes ---")
y_pred_tfidf = best_model_tfidf.predict(X_test)

tfidf_accuracy = accuracy_score(y_test, y_pred_tfidf)
print(f"Accuracy Score: {tfidf_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tfidf, target_names=['Ham', 'Spam']))

print("\nConfusion Matrix:")
cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
print(cm_tfidf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - TfidfVectorizer + MultinomialNB')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\n--- Email Spam Classification process completed ---")

# --- Optional: Test with a custom email/SMS ---
print("\n--- Optional: Custom Prediction Test ---")
custom_messages = [
    "Hey, how are you doing? Let's catch up later.", # Ham
    "URGENT! You have won a 1000 prize. Reply to claim now.", # Spam
    "Meeting at 3 PM tomorrow. Don't be late.", # Ham
    "Free entry to our exclusive offer. Text WIN to 888 now!" # Spam
]

print("\nPredictions using Best CountVectorizer + MultinomialNB model:")
for msg in custom_messages:
    prediction = best_model_cv.predict([msg])[0]
    label = 'Spam' if prediction == 1 else 'Ham'
    print(f"Message: '{msg}' -> Predicted: {label}")

print("\nPredictions using Best TfidfVectorizer + MultinomialNB model:")
for msg in custom_messages:
    prediction = best_model_tfidf.predict([msg])[0]
    label = 'Spam' if prediction == 1 else 'Ham'
    print(f"Message: '{msg}' -> Predicted: {label}")
```

-----

### How to Use the Application:

1.  **Download Dataset:**

      * Go to the [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset).
      * Download the `spam.csv` file.
      * Place `spam.csv` in the same directory as your Python script.

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

  * **Dataset:** We use the SMS Spam Collection Dataset. It's a classic text classification dataset with messages labeled as 'ham' (not spam) or 'spam'.
  * **Target Variable:** The 'label' column is converted to numerical `label_encoded` (0 for ham, 1 for spam) for the classification algorithms.
  * **Text Vectorization:** This is a crucial step in Natural Language Processing (NLP) to convert text data into numerical features that machine learning models can understand.
      * **`CountVectorizer`:** Converts a collection of text documents to a matrix of token counts (word frequencies).
      * **`TfidfVectorizer`:** Transforms text into TF-IDF (Term Frequency-Inverse Document Frequency) features. TF-IDF gives higher weight to words that are important in a document but not too common across all documents, making it good for identifying distinguishing terms.
  * **Naive Bayes (`MultinomialNB`):**
      * **Algorithm:** Multinomial Naive Bayes is a probabilistic classifier particularly well-suited for text classification tasks where features represent counts or frequencies (like word counts from `CountVectorizer` or TF-IDF values). It works based on Bayes' theorem with the "naive" assumption of conditional independence between features.
      * **Why it's good for spam:** It handles high-dimensional sparse data (common in text) very well and often performs surprisingly well with text data.
  * **Pipelines (`Pipeline`):**
      * Pipelines chain multiple processing steps (vectorization) with an estimator (classifier). This simplifies the workflow, ensures consistent preprocessing, and is essential for `GridSearchCV` to properly tune parameters across the entire workflow.
  * **Hyperparameter Tuning (`GridSearchCV`):**
      * `GridSearchCV` systematically searches for the best combination of hyperparameters for both the `vectorizer` (e.g., `max_df`, `min_df`, `ngram_range`) and the `classifier` (`alpha` for `MultinomialNB`). This helps to optimize model performance.
      * `max_df` and `min_df` control which words are included/excluded based on their document frequency.
      * `ngram_range` allows using sequences of words (e.g., "free money") instead of just single words.
      * `alpha` is a smoothing parameter in Naive Bayes that prevents zero probabilities for words not seen in the training data.
  * **Evaluation:**
      * `accuracy_score`: Overall percentage of correct predictions.
      * `classification_report`: Provides precision, recall, and F1-score for each class ("Ham" and "Spam"). These metrics are very important for imbalanced datasets (e.g., typically fewer spam emails than ham emails).
      * `confusion_matrix`: Visually shows true positives, true negatives, false positives, and false negatives.
