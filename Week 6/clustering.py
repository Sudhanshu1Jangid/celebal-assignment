import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier()
}

pipelines = {
    name: Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    for name, model in models.items()
}

results = {}

for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

# Display performance
pd.DataFrame(results).T.sort_values(by='f1_score', ascending=False)

svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(svc_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_svc_model = grid_search.best_estimator_
svc_preds = best_svc_model.predict(X_test)

print("Best Parameters for SVC:", grid_search.best_params_)
print(classification_report(y_test, svc_preds))

from scipy.stats import randint

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])

param_dist = {
    'rf__n_estimators': randint(50, 200),
    'rf__max_depth': randint(2, 20),
    'rf__min_samples_split': randint(2, 10),
    'rf__min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(rf_pipeline, param_distributions=param_dist, n_iter=30, cv=5, scoring='f1', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

best_rf_model = random_search.best_estimator_
rf_preds = best_rf_model.predict(X_test)

print("Best Parameters for Random Forest:", random_search.best_params_)
print(classification_report(y_test, rf_preds))

final_models = {
    "Tuned SVC": (best_svc_model, svc_preds),
    "Tuned Random Forest": (best_rf_model, rf_preds)
}

for name, (model, preds) in final_models.items():
    print(f"{name} - F1 Score: {f1_score(y_test, preds):.4f}")
