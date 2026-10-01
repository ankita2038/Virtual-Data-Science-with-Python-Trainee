# Week 4: Supervised Learning Model Implementation
# Iris Species Classification using Logistic Regression

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Split before fitting preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Build pipeline: scaling is fitted only on training folds/data
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# 4. Train model
model.fit(X_train, y_train)

# 5. Predict test data and evaluate
predictions = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, target_names=iris.target_names))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# 6. Cross-validation on training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
print("5-fold CV scores:", scores)
print("Mean CV accuracy:", scores.mean())
print("CV standard deviation:", scores.std())
