

## 🧠 **1. Basic Imports**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
```

---

## 📥 **2. Load or Create Sample Datasets**

```python
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target
```

---

## 🧪 **3. Split Dataset**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## ⚙️ **4. Data Preprocessing**

### Standardize Data

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### One-hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)
```

---

## 🔍 **5. Machine Learning Models**

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
```

### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
```

### Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

---

## 🎯 **6. Make Predictions and Evaluate Model**

### Prediction

```python
y_pred = model.predict(X_test)
```

### Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 🔧 **7. Cross-validation**

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
```

---

## 📊 **8. Grid Search (Hyperparameter Tuning)**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [3, 5, 7]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
```

---

## 💾 **9. Save and Load Model**

```python
import joblib

# Save
joblib.dump(model, 'model.pkl')

# Load
loaded_model = joblib.load('model.pkl')
```

---

## 📦 **10. Common Built-in Datasets**

```python
from sklearn.datasets import load_boston, load_digits, load_wine, fetch_20newsgroups
```

---
